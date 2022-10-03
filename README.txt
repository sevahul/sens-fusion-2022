To generate a pointcloud and a disparity image:
```
./run_naive.sh [output_file] [window_size] [n_jobs]
```
default values:
 output_file: output/output
 window_size: 3
 n_jobs: 0 (all threads)

camera parameters (`focal_length`, `baseline`, `dmin`) are read from `data/config.json` file


To visualize a pointcloud:
```
./visualize_cloud.py [points_file]
```
default values:
 points_file: output/output9.xyz


To calculate the similarity between disparities and disparitu groundtruth, run:
```
./compare_disparities.py
```

It will output the list of values for each metric. In list, values are compared for different window size (3, 5, 7, 9)
You can see the result saved in `output/metrics.txt`


REQUIREMENTS:
- jsoncpp library, libboost library.
INSTALLATION on Linux:
```
sudo apt-get install libjsoncpp-dev libopencv-dev python3-opencv
```
