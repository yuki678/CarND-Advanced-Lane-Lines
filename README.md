# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

The Project
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Installation
---
No specific installation is required. The code should run with python 3 with the imported libraries.
Just edit the last section of `P2.py` which defines input pictures or videos and run the command `python P2.py`.

Files
---
* P2.py - contains all the code
* camera_cal/  - The images for camera calibration are stored in the folder
* input_files/ - Input pictures and videos
  * `straight_linesX.jpg` are test inputs where the road is straight
  * `testX.jpg` are test inputs where the road is winding
  * `project_video.mp4` is the main target of this project
  * `challenge_video.mp4` is an extra (and optional) challenge under somewhat trickier conditions
  * `harder_challenge.mp4` video is another optional challenge and is brutal!
* output_files/ - Output pictures and videos are saved here
* advanced_lane_finding.md - The report of findings in this project

Licence
---
Everything in this codebase attributes to Udacity. Please refer to LICNSE file for the details.