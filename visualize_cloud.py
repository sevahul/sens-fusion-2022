#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import sys
import argparse
import os

if __name__ == "__main__":
    
    # print(dir(o3d.geometry.PointCloud))
    # exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest="input_file", nargs='?', default=os.path.join("output", "DP", "output"))
    args, unknown = parser.parse_known_args()
    ## define parameters
    fn = args.input_file
    if isinstance(fn, list):
        fn = fn[0]
    if ".xyz" not in fn:
        filename = "" + fn + ".xyz"
    min_z = 2000

    print(f"Visualising pointclout from the file {filename}...")
    ## load points
    points = np.loadtxt(filename)

    ## filter small z-coordinate
    points = points[np.where(points[:, 2] > min_z)]

    ## create PointCloud
    cl = o3d.geometry.PointCloud()
    cl.points = o3d.utility.Vector3dVector(points)

    ## filter points noise 
    voxel_down_pcd = cl.voxel_down_sample(voxel_size=0.02)
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=15, radius=5) # radius filter
    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=.2) # statistical filter

    ## visualize points
    #o3d.visualization.draw_geometries([cl], point_show_normal=True)
    print("Recompute the normal of the downsampled point cloud")
    
    downpcd = cl.voxel_down_sample(voxel_size=10)
    o3d.geometry.PointCloud.estimate_normals(
        downpcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=40,
                                                          max_nn=50))
    downpcd.orient_normals_towards_camera_location()
    o3d.visualization.draw_geometries([downpcd.voxel_down_sample(voxel_size=10)])


