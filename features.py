import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=True):
    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = [c[0] for c in point_colors_list]
    channel_2_vals = [c[1] for c in point_colors_list]
    channel_3_vals = [c[2] for c in point_colors_list]

    # Compute histograms:
    nbins=32
    bins_range=(0,255)
    ch1_hist = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    ch2_hist = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    ch3_hist = np.histogram(channel_3_vals, bins=nbins, range=bins_range)
    features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0])).astype(np.float64)
    return features / np.sum(features)


def read_normals(normal_cloud):
    return pc2.read_points(
        normal_cloud,
        field_names = ('normal_x', 'normal_y', 'normal_z'),
        skip_nans=True)


def compute_normal_histograms(normal_cloud):
    normals = read_normals(normal_cloud)
    norm_x_vals = [n[0] for n in normals]
    norm_y_vals = [n[1] for n in normals]
    norm_z_vals = [n[2] for n in normals]

    nbins = 360 / 16
    bins_range = (-1.0,1.0)
    nx_hist = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
    ny_hist = np.histogram(norm_y_vals, bins=nbins, range=bins_range)

    features = np.concatenate((nx_hist[0], ny_hist[0]))
    return features / np.sum(features)
