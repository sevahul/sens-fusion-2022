# STEREO VISION
## Analysis
The analysis is in the `analysis.pdf` file <br/>
You can also run `analysis.ipynb`, it should take several seconds if you haven't deleted anything from the repo. <br/>
## CLI tools
TLDR:
```
sudo apt-get install libopencv-dev python3-opencv
./mkrun.sh [options] # build and execute with default configurations
./visualize.py [ -f pointclod-file.xyz] # visualize a pointcloud
```
To only make the project:
```
./make.sh
```
To only run the project with default configurations (stereo extraction):
```
./run_stereo.sh [options]
```
or
```
./build/OpenCV_stereo [options]
```
To describe parameters:
```
./build/OpenCV_stereo --help
```
To visualize points from a specific file only:
```
./visualize_cloud.py [-f INPUT]
```
Parameters for `./build/OpenCV_stereo` are also set in `params.cfg` file, that is also explained in `./build/OpenCV_stereo --help` message.<br>
Parameters passed from the cli have more priority than the ones from the config file.
Check out `--help` option of the script to see the available metrics and usage. <br>

For the `naive` method, there is an old script to compare performance for different w_size:
```
./analysis/compare_disparities.py
```
It will output the list of values for each metric. In the list, values are compared for different window sizes (3, 5, 7, 9), you can see my result saved in `output/metrics.txt`.