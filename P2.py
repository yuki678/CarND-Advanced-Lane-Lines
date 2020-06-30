
# 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# 2. Apply a distortion correction to raw images.
# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# 5. Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip

def write_rgb_image(image, filepath):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Write the rgb image to filepath
    cv2.imwrite(filepath, rgb_image)

def calibrate_camera(nx=9, ny=6, chessboard_files='camera_cal/calibration*.jpg', sample_file='camera_cal/calibration1.jpg', is_chart=True):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(chessboard_files)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if is_chart:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Use the sample image to calibrate the camera
    img = cv2.imread(sample_file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Undistort the sample file (optional)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('calibration_wide/test_undist.jpg',dst)

    ## Visualization ##
    if is_chart:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=20)
        plt.show()

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )

    return dist_pickle

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def undist_perspective_transform(img, nx, ny, mtx, dist, offset, src, is_chart=True):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    # gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    
    # Grab the image shape
    img_size = (undist.shape[1], undist.shape[0])

    # Define the destination based on the offset
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                      [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    Minv = cv2.getPerspectiveTransform(dst, src)

    if is_chart:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 7))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(undist)
        ax2.set_title('Undistorted', fontsize=20)
        ax3.imshow(warped)
        ax3.set_title('Perspective Transformed', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # Return the resulting image and matrix
    return warped, M, Minv

def grad_color_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100), is_chart=True):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    ## Visualization ##
    if is_chart:
        # Plot the result
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 7))
        f.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20)

        ax2.imshow(color_binary)
        ax2.set_title('Stacked Threshold', fontsize=20)

        ax3.imshow(combined_binary, cmap='gray')
        ax3.set_title('Combined S Channel and Gradient Threshold', fontsize=12)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    return color_binary, combined_binary

def find_lane_pixels(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(img, leftx, lefty, rightx, righty, is_chart=True):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ## Visualization ##
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    if is_chart:
        # Colors in the left and right lane regions
        img[lefty, leftx] = [255, 0, 0]
        img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.figure(figsize=(12,7))
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(img)
        plt.show()

    return left_fit, right_fit, ploty, left_fitx, right_fitx

def measure_curvature(img, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''    
    # Define y-value where we want radius of curvature - the bottom of the image
    y_eval = img.shape[0]
    y_eval_ws = y_eval * ym_per_pix

    # Fit polynomials in meters without chart
    left_fit_ws, right_fit_ws, _, _, _ = fit_polynomial(img, leftx*xm_per_pix, lefty*ym_per_pix, rightx*xm_per_pix, righty*ym_per_pix, False)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_ws[0]*y_eval_ws + left_fit_ws[1])**2)**1.5) / np.absolute(2*left_fit_ws[0])
    right_curverad = ((1 + (2*right_fit_ws[0]*y_eval_ws + right_fit_ws[1])**2)**1.5) / np.absolute(2*right_fit_ws[0])
    
    # Calculation of center offset
    center_offset = img.shape[1] * xm_per_pix / 2 \
                    - ((left_fit_ws[0]*y_eval_ws**2 + left_fit_ws[1]*y_eval_ws + left_fit_ws[2]) \
                     + (right_fit_ws[0]*y_eval_ws**2 + right_fit_ws[1]*y_eval_ws + right_fit_ws[2])) / 2

    return left_curverad, right_curverad, center_offset

def make_out_img(org_img, warped_img, Minv, corners, ploty, left_fitx, right_fitx, r_curvature, c_offset, is_chart=True):
    # Create a blank image to draw lines
    warped_blank = np.zeros_like(warped_img).astype(np.uint8)
    #warped_color = np.dstack((warped_blank, warped_blank, warped_blank))

    img_size = (org_img.shape[1], org_img.shape[0])
    # Recast the x and y points for cv2.fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warped_blank, [pts], (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarp = cv2.warpPerspective(warped_blank, Minv, img_size)
    
    # Combine the result with the original image
    result = cv2.addWeighted(org_img, 1, unwarp, 0.3, 0)
    
    # Write text on image
    font = cv2.FONT_HERSHEY_TRIPLEX    
    curvature_text = "Radius of Curvature = " + str(round(r_curvature, 0)) + " (m)"
    cv2.putText(result, curvature_text, (30, 60), font, 1, (0,255,0), 2)

    if c_offset > 0:
        offset_text = "Vehicle is {:.2f} (m) right of center".format(np.abs(c_offset))
    elif c_offset < 0:
        offset_text = "Vehicle is {:.2f} (m) left of center".format(np.abs(c_offset))
    elif c_offset == 0:
        offset_text = "Vehicle is at the center"
    cv2.putText(result, offset_text, (30, 90), font, 1, (0,255,0), 2)

    if is_chart:
        plt.figure(figsize=(12,7))
        plt.imshow(result)    
        plt.show()

    return result

def process_image(image, is_chart=False):
    ## Define variables ##
    
    # Inner corners of chequer boards
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # For source points, grabbing the outer four detected corners
    offset = 200
    corners = np.float32([[580, 460], [705, 460], [1042, 675], [273, 675]])

    # 1. Calibration calculation
    # required only for the first time. The dump file can be loaded from the second time.
    #dist_pickle = calibrate_camera(nx, ny)
    dist_pickle = pickle.load( open( "calibration_wide/wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # 2. Unwarping in an image
    top_down, M, Minv = undist_perspective_transform(image, nx, ny, mtx, dist, offset, corners, is_chart)

    # 3. Gradient and Color 
    result_color, result_combined = grad_color_binary(top_down, (170, 255), (20, 100), is_chart)

    # 4. Find lanes
    leftx, lefty, rightx, righty, lane_img = find_lane_pixels(result_combined)

    # 5. Fit polynomials
    left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(lane_img, leftx, lefty, rightx, righty, is_chart)

    # 6. Calculate the radius of curvature in meters for both lane lines
    left_r, right_r, c_offset = measure_curvature(lane_img, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)
    r_curvature = (left_r + right_r) // 2

    # 7. Generate the output picture
    result = make_out_img(image, lane_img, Minv, corners, ploty, left_fitx, right_fitx, r_curvature, c_offset, is_chart)

    return result

# Test Image
filename = "test2.jpg"
image = cv2.imread("input_files/" + filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = process_image(image, True)
write_rgb_image(result, "output_files/" + filename)

# # Test Video
# filename = "project_video.mp4"
# clip1 = VideoFileClip("input_files/" + filename)
# clip1_output = clip1.fl_image(process_image)
# clip1_output.write_videofile("output_files/" + filename, audio=False)