#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import subprocess

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd

def SSIM(img1, img2):
    img1_float = img_as_float(img1)
    img2_float = img_as_float(img2)
    ssim_score = ssim(img1_float, img2_float, data_range=img1_float.max() - img1_float.min())
    return ssim_score

if __name__ == "__main__":
    
    # read gt disparity image
    output_folder_list = ["output", "DP"]
    output_folder = os.path.join(*output_folder_list)
    output_template_name = "output"
    data_folder = "data"
    img_gt_path = os.path.join(data_folder, "disp1.png")
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)

    # list of previously calculated disparity images for each window size 
    
    lambdas = np.arange(1, 11, 2)
    w_sizes = np.arange(1, 7, 2)
    
    columns = pd.MultiIndex.from_arrays([["lambda"]*len(lambdas), lambdas])
    index = pd.MultiIndex.from_arrays([["window-size"]*len(w_sizes), w_sizes])
    df = pd.DataFrame(columns=columns, index=index)
    
    print(df)
    for l in lambdas:
        for w in w_sizes: 
            filename = os.path.join(output_folder, output_template_name + f"_l{l}" + f"_w{w}" + f"_DP.png")
            if not os.path.exists(filename):
                cmd = ['./build/OpenCV_stereo', '-H1', f'-l{l}', f'-w{w}']
                subprocess.Popen(cmd).wait()
            
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            df.loc[('window-size', w), ('lambda', l)] = SSIM(image, img_gt)
            print(df, "\n")
    exit(0)
    # calculate metrics
    for my_img in my_images:
        img_path = os.path.join(output_folder,output_subfolder, my_img)
        img_my = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        nccs.append(NCC(img_my, img_gt))
        mses.append(MSE(img_my, img_gt))
        ssims.append(SSIM(img_my, img_gt))
    print("w_size:\t   3    5    7    9")
    print("NCC:\t", np.round(nccs, 2))
    print("MSE:\t", np.round(mses, 2))
    print("SSIM:\t", np.round(ssims, 2))
 
