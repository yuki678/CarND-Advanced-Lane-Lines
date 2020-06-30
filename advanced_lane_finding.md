# Advanced Lane Finding Project
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

All the code is included in `P2.py`

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./input_files/test2.jpg "Road Transformed"
[image3]: ./output_files/test2_warped.png "Warp Example"
[image4]: ./output_files/test2_identify.png "Identify Example"
[image5]: ./output_files/test2_polynomial.png "Fit Visual"
[image6]: ./output_files/test2_plotted_back.png "Output"
[video1]: ./project_video.mp4 "Video"

---

## Camera Calibration

#### How the camera matrix and distortion coefficients are computed

The code for this step is contained in `calibrate_camera` function. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---

## Pipeline (single images)
To demonstrate how the process was applied to an image, I will use this image as an example.
![alt text][image2]

### 1. Distortion-correction and perspective transforms

The code for this step is contained in `undist_perspective_transform` function. 

First, I applied the distortion correction to the original image using the output of Camera Calibration, the camera calibration and distortion coefficients.

Then, I converted the perspective. To determine the source (`src`) points by carefully examining the input and then coded the destination (`dst`) points as follows:

```python
dst = np.float32([[offset, 0], 
                  [img_size[0]-offset, 0], 
                  [img_size[0]-offset, img_size[1]], 
                  [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 460      | 200, 0        |
| 705, 460      | 1080, 0       |
| 1042, 675     | 1080, 720     |
| 273, 675      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]


### 2. Color transforms and gradients to create a thresholded binary image.

The code for this step is contained in `grad_color_binary` function. 

I used a combination of color and gradient thresholds to generate a binary image. First, Sobel function was applied to calculate the gradient in x direction. The x derivative is then filtered to make the binary output. Within the threshold is considered as 1. Then, convert the image to HLS and s channel was taken. Here, again the threshold was applied to generate a binary output. Finally, combined the result to mark 1 when either is 1.

Here's an example of my output for this step. 

![alt text][image4]

### 3. Identify lane-line pixels and fit their positions with a polynomial

The code for this step is contained in `find_lane_pixels` function and `fit_polynomial` function.

First, I calculated the histogram and find two peaks for the left lane and right lane.
Then, created the first window around the peaks, followed by sliding windows of focus by taking the mean of non-zero pixels. Within each window, took nonzero pixels to create x and y values for left and right lane respectively.

I passed the outputs to `fit_polynomial` to fit my lane lines with a 2nd order polynomial like this:
```python
# 2nd Order Polinomial
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values
ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

![alt text][image5]

### 4. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

The code for this step is contained in `measure_curvature` function. 

First, consider the ratio between pixel and actual distance in meters as follows.
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
In order to calculate the radius of the curveture, first fit the polynomial in meter and then applied the below formula to calculate for each lane. Took the average as the final radius value.
Here, _ws stands for actual world space value (in meter).

```python
# Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_ws[0]*y_eval_ws + left_fit_ws[1])**2)**1.5) / np.absolute(2*left_fit_ws[0])
right_curverad = ((1 + (2*right_fit_ws[0]*y_eval_ws + right_fit_ws[1])**2)**1.5) / np.absolute(2*right_fit_ws[0])
```

Offset from the center was calculated as the deviation of the mean of each polynomial at the bottom from the image center as follows.
```python
# Calculation of center offset
center_offset = img.shape[1] * xm_per_pix / 2 \
                - ((left_fit_ws[0]*y_eval_ws**2 + left_fit_ws[1]*y_eval_ws + left_fit_ws[2]) \
                 + (right_fit_ws[0]*y_eval_ws**2 + right_fit_ws[1]*y_eval_ws + right_fit_ws[2])) / 2
```

### 5. Plot the result back down onto the road

The code for this step is contained in `make_out_img` function. 

I used OpenCV's fillPoly() function to fill the area within left and right polynomials. Then, convert back the perspective to the original image space using Minv, which is the inverse matrix of the perspective transformation. Finally, combine it with the original imange and added text about the curvature and offset.

Here is an example of my result on a test image:
![alt text][image6]

---

## Pipeline (video)

### Apply the pipeline to the final video

Here's a [link to my video result](./output_files/project_video.mp4)

My pipeline performed reasonably well on the entire project video without catastrophic failures, while there are some improvements can be made for whilwobbly lines.

---

## Discussion

### Any problems or issues that could be improved further

One of the major issue of this algorithm is the use of sliding window to each video frame. In some frames, the lanes are not so obvious within the frame image, but it can be easily identified when considering the series of images - i.e. consider the lane position of the previous frames to estimate rough position of the lane as it will not suddenly jump.
It could be also improved by using sensor information like acceleration in combination with the previous frames.
