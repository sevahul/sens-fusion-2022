#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import sys

if __name__ == "__main__":

    ## define parameters
    filename = "output/output.xyz"
    min_z = 2000

    if len(sys.argv) > 1:
        filename = sys.argv[1]

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
    o3d.visualization.draw_geometries([cl], point_show_normal=True)


