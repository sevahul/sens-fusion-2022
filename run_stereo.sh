#!/usr/bin/env bash
./build/OpenCV_stereo "$@"
# status=1
# #output=$(./build/OpenCV_stereo "$@" && status=$? | tee /dev/stderr)
# output=$(echo "haha" && status=$? | tee /dev/stderr)
# #status=$?
# echo $status
# exit 0
# [ $status -eq 0 ] && ./visualize_cloud.py -f $output
