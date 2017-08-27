#!/usr/bin/env python

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

from capture_features import extract_features


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster



def voxel_downsampling(c, leaf_size=0.002):
    result = c.make_voxel_grid_filter()
    result.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return result.filter()


def passthrough_filter(c, axis_min=0.77, axis_max=1.1, filter_axis='z'):
    result = c.make_passthrough_filter()
    result.set_filter_field_name(filter_axis)
    result.set_filter_limits(axis_min, axis_max)
    return result.filter()


def ransac(cloud, max_distance=0.01):
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    return cloud.extract(inliers, negative=False), cloud.extract(inliers, negative=True)


def perform_clustering(cloud):
    cloud_xyz = XYZRGB_to_XYZ(cloud)
    tree = cloud_xyz.make_kdtree()
    ec = cloud_xyz.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    return ec.Extract()


def visualize_clusters(cloud, cluster_indices):
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


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    c = ros_to_pcl(pcl_msg)
    c = voxel_downsampling(c)
    c = passthrough_filter(c)
    table_cloud, object_cloud = ransac(c)
    cluster_indices = perform_clustering(object_cloud)
    cluster_cloud = visualize_clusters(object_cloud, cluster_indices)
    detected_objects, detected_objects_labels = detect_objects(object_cloud, cluster_indices)

    pcl_objects_pub.publish(pcl_to_ros(object_cloud))
    pcl_table_pub.publish(pcl_to_ros(table_cloud))
    pcl_cluster_pub.publish(pcl_to_ros(cluster_cloud))
    detected_objects_pub.publish(detected_objects)
# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)

        # Grab the points for the cluster

        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects

if __name__ == '__main__':
    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize the ros node:
    rospy.init_node('object_recognition', anonymous=True)

    # Subscriber:
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Publishers:
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
