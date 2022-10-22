TLDR:
```
sudo apt-get install libopencv-dev python3-opencv
./mkrun.sh [options] #build, extract and visualize with default configurations
./estimate_method.py
```
To only make the project:
```
./make.sh
```
To only run the project with default configurations (stereo extraction + visualization):
```
./run_stereo.sh [options]
```
To run stereo extraction only:
```
./build/OpenCV_stereo [options]
```
To describe parameters:
```
./build/OpenCV_stereo --help
```
To visualize points from a specific file only:
```
./visualize_cloud.py [-o INPUT]
```
Parameters for `./build/OpenCV_stereo` are also set in `params.cfg` file, that is also explained in `./build/OpenCV_stereo --help` message.<br>
Parameters passed from the cli have more priority than the ones from the config file.
To estimate the `DP` method performance versus `naive` method with different values of `window-size` and `lambda` parameters, run:
```
./estimate_method.py
```
Check out `--help` option of the script to see the available metrics and usage. <br>
My results can be seen in the `output/comparison.txt` file. <br>
You can optionally change the values that you want to check inside the script. <br>

For the `naive` method, there is another old script:
```
./compare_disparities.py
```
It will output the list of values for each metric. In the list, values are compared for different window sizes (3, 5, 7, 9), you can see my result saved in `output/metrics.txt`.