#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import subprocess

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import os
import pandas as pd
import argparse

def NCC(img1, img2): 
    img_my_ar = img1.flatten()
    img_gt_ar = img2.flatten()
    img_my_ar_norm = (img_my_ar - img_my_ar.mean())/np.linalg.norm(img_my_ar)
    img_gt_ar_norm = (img_gt_ar - img_gt_ar.mean())/np.linalg.norm(img_gt_ar)
    cor = np.corrcoef(img_gt_ar_norm, img_my_ar_norm)
    return cor[1,0]

def MSE(img1, img2):
    img1_float = img_as_float(img1)
    img2_float = img_as_float(img2)
    mse_score = mean_squared_error(img1_float, img2_float)
    return mse_score
 
def SSIM(img1, img2):
    img1_float = img_as_float(img1)
    img2_float = img_as_float(img2)
    ssim_score = ssim(img1_float, img2_float, data_range=img1_float.max() - img1_float.min())
    return ssim_score

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric', default="SSIM", help="Comparison metric. Should be 'SSIM', 'MSE', or 'NCC'")
    args = parser.parse_args()
    metric_name = args.metric.lower()
    metric = None
    if metric_name == 'ssim':
        metric = SSIM
    elif metric_name == 'ncc':
        metric = NCC
    elif metric_name == 'mse':
        metric = MSE
    else:
        print(f"Unkown metric {metric}! Use --help to find the right usage")
    print(metric_name)

    # read gt disparity image
    output_folder_list = ["output", "DP"]
    output_folder = os.path.join(*output_folder_list)
    output_template_name = "output"
    data_folder = "data"
    img_gt_path = os.path.join(data_folder, "disp1.png")
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)

    # list of previously calculated disparity images for each window size 
    
    lambdas = np.arange(1, 11, 2)
    w_sizes = np.arange(1, 11, 2)
    
    columns = pd.MultiIndex.from_arrays([["lambda"]*len(lambdas) + ["naive"], list(lambdas) + [""] ])
    index = pd.MultiIndex.from_arrays([["window-size"]*len(w_sizes), w_sizes])
    df = pd.DataFrame(columns=columns, index=index)
    
    print(df)
    for l in lambdas:
        for w in w_sizes[:-2]: 
            filename = os.path.join(output_folder, output_template_name + f"_l{l}" + f"_w{w}" + f"_DP.png")
            if not os.path.exists(filename):
                cmd = ['./build/OpenCV_stereo', '-H1', f'-l{l}', f'-w{w}']
                subprocess.Popen(cmd).wait()
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            df.loc[('window-size', w), ('lambda', l)] = metric(image, img_gt)
            print(f"{metric_name.upper()}:\n", df, "\n")
     
    for w in w_sizes:
        output_folder = os.path.join("output", "naive")
        filename = os.path.join(output_folder, output_template_name + f"_w{w}" + f"_naive.png")
        if not os.path.exists(filename): 
                cmd = ['./build/OpenCV_stereo', '-H1', f'-mnaive', f'-w{w}', '-o', 'output/naive/output']
                subprocess.Popen(cmd).wait()
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        df.loc[('window-size', w), ('naive', "")] = metric(image, img_gt)
        print(f"{metric_name.upper()}:\n", df, "\n")
    exit(0)
