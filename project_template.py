#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

from std_msgs.msg import Int32, String
from geometry_msgs.msg import Pose


def make_ros_int32(a):
    """
    Construct a ROS Int32 object from an int.

    Arguments:
        a: an integer

    Returns:
        ROS Int32 object with value a
    """
    ros_int32 = Int32()
    ros_int32.data = a
    return ros_int32


def make_ros_string(s):
    """
    Construct a ROS String object from a python string.

    Arguments:
        s: a python string

    Returns:
        ROS String object with value s
    """
    ros_string = String()
    ros_string.data = s
    return ros_string


def make_ros_pose(x, y, z):
    """
    Construct a ROS Pose object from .

    Arguments:
        x,y,z: position coordinates

    Returns:
        ROS Pose object with position (x,y,z)
    """
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    return pose


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = int(test_scene_num.data)
    yaml_dict["arm_name"]  = str(arm_name.data)
    yaml_dict["object_name"] = str(object_name.data)
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def extract_features(cloud):
    """
    Extract SVM features from point cloud.

    Parameters:
        cloud: ROS point cloud data.

    Returns:
        Normalized vector of color and normal histogram features.
    """
    color_histograms = compute_color_histograms(cloud, using_hsv=True)
    normals = get_normals(cloud)
    normal_histograms = compute_normal_histograms(normals)
    return np.concatenate((color_histograms, normal_histograms))


def voxel_downsampling(c, leaf_size=0.002):
    """
    Perform voxel downsampling on point cloud data.

    Parameters:
        c: pcl point cloud
        leaf_size: leaf size for downsampling.

    Returns:
        Downsampled point cloud.
    """
    result = c.make_voxel_grid_filter()
    result.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return result.filter()


def outlier_filter(c, mean_k=50, std_dev_mul_thresh=0.05):
    """
    Perform outlier filtering.

    Parameters:
        c:  pcl point cloud
        mean_k, std_dev_mul_thresh: Filter parameters

    Returns:
        Filtered pcl point cloud.
    """
    of = c.make_statistical_outlier_filter()
    of.set_mean_k(mean_k)
    of.set_std_dev_mul_thresh(std_dev_mul_thresh)
    return of.filter()


def passthrough_filter(c, axis_min=0.60, axis_max=1.1, filter_axis='z'):
    """
    Perform passthrough filtering.

    Parameters:
        c: pcl point cloud
        axis_min, axis_max: min/max values for filtering along axis.
        filter_axid: axis for filtering

    Returns:
        Filtered pcl point cloud.
    """
    result = c.make_passthrough_filter()
    result.set_filter_field_name(filter_axis)
    result.set_filter_limits(axis_min, axis_max)
    return result.filter()


def ransac(cloud, max_distance=0.01):
    """
    Perfomm RANSAC filering.

    Parameters:
        cloud: pcl point cloud
        max_distance: max distance from plane.

    Returns:
        inliners: pcl point cloud consisting of inliners
        outliers: pcl point cloud conisting of outliers
    """
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    return cloud.extract(inliers, negative=False), cloud.extract(inliers, negative=True)


def perform_clustering(cloud):
    """
    Perform euclidean clustering.

    Parameters:
        cloud: pcl point cloud

    Returns:
        Array of point cloud indices of clusters.
    """
    cloud_xyz = XYZRGB_to_XYZ(cloud)
    tree = cloud_xyz.make_kdtree()
    ec = cloud_xyz.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    return ec.Extract()


def visualize_clusters(cloud, cluster_indices):
    """
    Visualize clusters.

    Parameters:
        cloud: Original pcl point cloud.
        cluster_indices: array of cluster indices

    Returns:
        Pcl cloud with colorized clusters.
    """
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        color = rgb_to_float(cluster_color[j])
        for idx in indices:
            p = cloud[idx]
            color_cluster_point_list.append([p[0], p[1], p[2], color])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud


def detect_objects(object_cloud, cluster_indices):
    """
    Classify object with an SVM classifier.

    Parameters:
        object_cloud: pcl point cloud of objects to classify
        cluster_indices: array of cluster indices

    Returns:
        detected_objects: Array of objects of type DetectedObject
        detected_objects_labels: Array of object labels
    """
    detected_objects = []
    detected_objects_labels = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = object_cloud.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        features = extract_features(ros_cluster)

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(features.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(object_cloud[pts_list[0]])[:-1]
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    return detected_objects, detected_objects_labels


def get_object_for_label(object_list, label):
    """
    Get DetectedObject object with certain label from list of detected objects.

    Parameters:
        object_list: array of DetectedObject
        labe: label to find

    Returns:
        Object of type DetectedObject that has the label or None if it was not found.
    """
    for do in object_list:
        if do.label == label:
            return do
    return None


def calc_centroid(item):
    """
    Calc centroid of a model point cloud.

    Parameters:
        item: Object of type DetectedObject

    Returns:
        np.array with (x,y,z) position of centroid of the object.
    """
    points_arr = ros_to_pcl(item.cloud).to_array()
    return np.mean(points_arr, axis=0)[:3]


def create_pick_pose(item):
    """
    Create a pick pose for an object.

    Parameters:
        item: Object of type DetectedObject

    Returns:
        ROS Pose object of pick position.
    """
    centroid = calc_centroid(item)
    return make_ros_pose(centroid[0], centroid[1], centroid[2])


def create_place_pose(item_group):
    """
    Create a place pose for an item group.

    Parameters:
        item_group: string of object group 'red' or 'green'

    Return:
        ROS Pose object of place position.
    """
    px = 0
    py = 0.71 if item_group == 'red' else -0.71
    pz = 0.605
    return make_ros_pose(px,py,pz)


def select_arm(item_group):
    """
    Select one of the robot arms based on item group.

    Parameters:
        item_group: string of object group 'red' or 'green'

    Returns:
        ROS String object of robot arm name ('left' or 'right').
    """
    arm_name = 'left' if item_group == 'red' else 'right'
    return make_ros_string(arm_name)


def turn_pr2(start_angle, end_angle):
    """
    Turn robot.

    Parameters:
        start_angle, end_angle: start/end angle of the turning maneuver.
    """
    joint_controller_pub.publish(end_angle)
    rospy.sleep(abs(end_angle - start_angle) / np.pi * 32.0)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Filter the point cloud:
    c = ros_to_pcl(pcl_msg)
    c = outlier_filter(c)
    c = voxel_downsampling(c)
    c = passthrough_filter(c, 0.6, 1.1, 'z')
    c = passthrough_filter(c, -0.5, 0.5, 'y')
    table_cloud, object_cloud = ransac(c)

    # Perform clustering:
    cluster_indices = perform_clustering(object_cloud)
    cluster_cloud = visualize_clusters(object_cloud, cluster_indices)

    # Perform object classification:
    detected_objects, detected_objects_labels = detect_objects(object_cloud, cluster_indices)

    # Publish:
    pcl_objects_pub.publish(pcl_to_ros(object_cloud))
    pcl_table_pub.publish(pcl_to_ros(table_cloud))
    pcl_cluster_pub.publish(pcl_to_ros(cluster_cloud))
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    dict_list = []
    object_list_param = rospy.get_param('/object_list')
    test_scene_num = make_ros_int32(rospy.get_param('/test_scene_num'))

    turn_pr2(0.0, np.pi/2)
    turn_pr2(np.pi/2, -np.pi/2)
    turn_pr2(-np.pi/2, 0.0)

    for object_dict in object_list_param:
        item_name = object_dict['name']
        item_group = object_dict['group']

        item = get_object_for_label(object_list, item_name)
        if  item == None:
            continue

        arm_name = select_arm(item_group)
        item_name = make_ros_string(item_name)
        pick_pose = create_pick_pose(item)
        place_pose = create_place_pose(item_group)
        dict_list.append(make_yaml_dict(test_scene_num, arm_name, item_name, pick_pose, place_pose))

        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(
                test_scene_num,
                item_name,
                arm_name,
                pick_pose,
                place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    yaml_filename = "object_%d.yaml" % rospy.get_param('/test_scene_num')
    print("Writing to %s" % yaml_filename)
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize the ros node:
    rospy.init_node('object_recognition', anonymous=True)

    # Subscriber:
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Publishers:
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    joint_controller_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
