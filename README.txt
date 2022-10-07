TLDR:
```
sudo apt-get install libjsoncpp-dev libopencv-dev python3-opencv
mkdir build
cd build
cmake ..
cd ..
./make.sh
# optionally: edit data/config.json file
./run_stereo.sh
./visualize_cloud.py
./compare_disparities.py
```



To generate a pointcloud and a disparity image:
```
./run_stereo.sh [output_file [-w window_size] [-j n_jobs]]
```

default values:
  -j: 16

other default values are in `data/config.json` file
parameters passed in the script from cli have more priority then the ones in json
 

To visualize a pointcloud:
```
./visualize_cloud.py [points_file]
```
default values:
 points_file: output/output.xyz


To calculate the similarity between disparities and disparities groundtruth, run:
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

Script `make.sh` is used to recompile the code.
Script `mkrun.sh` is used to make and run code with parameters from the config.

For "method" field of the config, either "DP" of "naive" value have to be used.
