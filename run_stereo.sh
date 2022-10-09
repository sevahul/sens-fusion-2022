#!/usr/bin/env bash

./build/OpenCV_stereo "$@" && ./visualize_cloud.py "$@"
