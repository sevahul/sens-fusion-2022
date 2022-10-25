import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import cv2
import pandas as pd
import json
import time
# import custom metrics
from analysis.compare_disparities import NCC, MSE, SSIM

# define metrics, datasets, methods and parameters ranges
metrics_dict = {
    "SSIM": lambda x, y: SSIM(x, y, True),
    "MSE": lambda x, y: MSE(x, y, True),
    "NCC": lambda x, y: NCC(x, y)
}
datasets = next(os.walk('data'))[1]
methods = ["DP", "naive"]
lambdas = list(range(1, 10, 2)) + [20, 50]
w_sizes = [1, 3, 5, 7, 9]
padding = 30

## reading and writing existing cached results

# for metrics values
metrics_dict_path = os.path.join("analysis", "metrics.json")
def read_metrics():
    if os.path.isfile(metrics_dict_path):
        with open(metrics_dict_path) as f:
            return json.load(f)
    else:
        return dict() 

def write_metrics(metrics_dict):
    with open(metrics_dict_path, 'w') as convert_file:
        convert_file.write(json.dumps(metrics_dict, indent=4))

# for execution times
times_dict_path = os.path.join("analysis", "times.json")
def read_times():
    if os.path.isfile(times_dict_path):
        with open(times_dict_path) as f:
            return json.load(f)
    else:
        return dict() 

def write_times(times_dict):
    with open(times_dict_path, 'w') as convert_file:
        convert_file.write(json.dumps(times_dict, indent=4))

## execution tools

# get output png full name for given parameters
def get_full_name(Dataset, Algo, w_size=1, l=9):
    filename = f"output"
    if Algo == "DP": filename += f"_l{l}"
    filename += f"_w{w_size}_{Algo}.png"
    file_full_name = os.path.join("output", Algo, Dataset, filename)
    return file_full_name, os.path.exists(file_full_name)

# get gt disparity image for a Dataset
def get_img_gt(Dataset):
    data_folder = os.path.join("data", Dataset)
    img_gt_path = os.path.join(data_folder, "disp1.png")
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
    return img_gt

# run algorithm in case it wasn't run or the execution time is not benchmarked
def run_algo(Dataset, Algo, w_size=1, l=1):
    file_full_name, _ = get_full_name(Dataset, Algo, w_size, l)
    output_folder = os.path.join("output", Algo, Dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    times = read_times()
    time_keys = times.keys()
    if not os.path.isfile(file_full_name) or file_full_name not in time_keys:
        print(f"{file_full_name} does not exist. Computing disparity")
        command = ['./build/OpenCV_stereo',
                                        f'data/{Dataset}/view0.png', 
                                        f'data/{Dataset}/view1.png',
                                        f'output/{Algo}/{Dataset}/output',
                                        f'-l{l}', f'-w{w_size}', "-Htrue", "-ntrue", f"-m{Algo}"]
        start = time.time()
        print(f"executing command {' '.join(command)}")
        process = subprocess.Popen(command)
        stdout, stderr = process.communicate()
        end = time.time()
        execution_time = end - start
        times[file_full_name] = np.round(execution_time, 2)
        write_times(times)
    return file_full_name


# comparing the results with the groundtruth using the existing metrics (run stereo if not cached)
def compare_to_gt(Dataset, Algo, w_size=1, l=1):
    all_metrics = read_metrics()
    filename, _ = get_full_name(Dataset, Algo, w_size, l)
    if filename in all_metrics.keys():
        return all_metrics[filename]

    run_algo(Dataset, Algo, w_size=w_size, l=l)
    img_gt = get_img_gt(Dataset)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    metrics = {}
    for metric_name, metric in metrics_dict.items():
            metrics[metric_name] = metric(img[:, padding:], img_gt[:, :-padding])
    all_metrics[filename] = metrics
    write_metrics(all_metrics)
    return metrics


## define functions to get the metrics dataframes depending on a dataset

# and depending on lambda
def get_metrics_lambda(Dataset):
    Algo = "DP"

    metrics = pd.DataFrame()
    for l in lambdas:
        metrics_local = compare_to_gt(Dataset, Algo, l=l)
        for metric_name in metrics_dict.keys():
            metrics.loc[l, metric_name] = metrics_local[metric_name]
    return metrics

# and depending on method
def get_metrics_method(Dataset):
    metrics = pd.DataFrame()
    l = 9

    for Algo in methods:
        method = Algo
        if Algo == "DP": method += "(ws=1, l=9)"
        elif Algo == "naive": method += "(ws=9)"
        w_size = 1 if Algo == "DP" else 9

        metrics_local = compare_to_gt(Dataset, Algo, w_size = w_size, l=l)
        for metric_name in metrics_dict.keys():
            metrics.loc[method, metric_name] = metrics_local[metric_name]
    return metrics

# and depending on w_size, Algo
def get_metrics_w_size(Dataset, Algo):
    l = 9
    metrics = pd.DataFrame()
    for w_size in w_sizes:
        metrics_local = compare_to_gt(Dataset, Algo, w_size = w_size, l=l)
        for metric_name in metrics_dict.keys():
            metrics.loc[w_size, metric_name] = metrics_local[metric_name]
    return metrics
# and depending on w_size for DP
def get_metrics_w_size_DP(Dataset):
    return get_metrics_w_size(Dataset, "DP")
# and depending on w_size for naive
def get_metrics_w_size_naive(Dataset):
    return get_metrics_w_size(Dataset, "naive")


# get average metrics across Datasets for a given metric extraction function
def get_avg_metrics(get_metrics_func):
    avg_metrics = None
    for Dataset in datasets:
        if avg_metrics is None:
            avg_metrics = get_metrics_func(Dataset)
        else:
            avg_metrics += get_metrics_func(Dataset)
    # averaging
    avg_metrics = avg_metrics/len(datasets)
    return avg_metrics


# visualize image diff for a given dataset
def display_image_diff(Dataset):
    f, ax = plt.subplots(1, len(methods))
    f.set_figheight(10)
    f.set_figwidth(30)
    for i, Algo in enumerate(methods):
        l = 9

        w = 1 if Algo == "DP" else 9
        img_gt = get_img_gt(Dataset)
        gt_normed = cv2.normalize(img_gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        output_folder = os.path.join("output", Algo, Dataset)
        file_full_name = run_algo(Dataset, Algo, w, l)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image = cv2.imread(file_full_name, cv2.IMREAD_GRAYSCALE)
        orig_normed = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
        image_diff = orig_normed[:, padding:] - gt_normed[:, :-padding]
        ax[i].imshow(image_diff * 255,cmap='gray', vmin=0, vmax=255)
        ax[i].set_title(Algo)
    plt.suptitle(f"Dataset {Dataset} diff")
    plt.show()


# get execution time from the saved time json file
def get_execution_time(Dataset = None, Algo=None, w_s=None, l=None):
    times = read_times()
    times = {key[:-4]: value for key, value in times.items()}
    if Algo is not None: times = {key: value for key, value in times.items() if key.split("/")[1] == Algo}
    if Dataset is not None: times = {key: value for key, value in times.items() if key.split("/")[2] == Dataset}
    if w_s is not None: times = {key: value for key, value in times.items() if key.split("_")[-2] == f"w{int(w_s)}"}
    if Algo == "DP" and l is not None:
        times = {key: value for key, value in times.items() if key.split("_")[1] == f"l{int(l)}"}
    return times

# get execution time for a given Dataset depending on method and window_size 
def get_time_method_ws(Dataset):
    exec_times_ws = pd.DataFrame()
    for Algo in methods:
        for ws in w_sizes:
            e_time = list(get_execution_time(Algo=Algo, Dataset=Dataset, w_s=ws, l=9.0).values())[0]
            exec_times_ws.loc[ws, Algo] = e_time
    return exec_times_ws

# get execution time for a given Dataset depending on lambda (DP method)
def get_time_DP_lambda(Dataset):
    Algo="DP"
    exec_times_ws = pd.DataFrame()
    for l in lambdas:
        for ws in w_sizes:
            e_time = list(get_execution_time(Algo="DP", Dataset=Dataset, w_s=1, l=l).values())[0]
            exec_times_ws.loc[ws, Algo] = e_time
    return exec_times_ws