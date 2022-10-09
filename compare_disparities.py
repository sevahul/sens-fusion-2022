#!/usr/bin/env python3

import cv2
import numpy as np
import sys

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os

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
    
    # read gt disparity image
    output_folder = "output"
    data_folder = "data"
    output_subfolder = "naive"
    img_gt_path = os.path.join(data_folder, "disp1.png")
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)

    # list of previously calculated disparity images for each window size
    my_images = ["output3_naive.png",
                 "output5_naive.png", 
                 "output7_naive.png", 
                 "output9_naive.png"]
    nccs = []
    mses = []
    ssims = []
    
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
 
