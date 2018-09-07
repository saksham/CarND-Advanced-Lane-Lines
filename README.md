## Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![identifying-lane-lines-youtube-video](http://img.youtube.com/vi/bdHtsbaUso8/0.jpg)](http://www.youtube.com/watch?v=bdHtsbaUso8 "Finding Lane Lines for Autonomous Driving")

## Table of Contents
* [Project Description](#the-project)
* [What's in this repo](#what's-in-this-repo)
* [Camera calibration](#camera-calibration)
* [Pipeline (single images)](#pipeline-(single-images))
* [Pipeline (video)](#pipeline-(video))
* [Discussion](#discussion) 

## The Project

The goals/steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

This README is structured around the [Rubric](https://review.udacity.com/#!/rubrics/571/view) points for the project.

## What's in this Repo

```
├── example.ipynb: Main notebook for exploration and showcasing
├── data: contains calibration images, example images and videos
├── output: result of processing
├── scripts: helper scripts for running/testing code snippets
└── src: all the code is here
```

Additionally, there is [environments.yml](environment.yml) file which can be used to set up
local environment to run the notebook and other scripts. For more details, refer to [this repo from
Udacity](https://github.com/udacity/CarND-Term1-Starter-Kit.git).

```bash
conda env create -f environment.yml
```

---

[//]: # (Image References)
[distortion-correction]: ./data/output/images/calibration/distortion-correction.png "Distortion correction applied to chess board image"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


<a name="camera-calibration"></a>
### Camera Calibration

The code for camera calibration can be found in the `calibrate` function inside [src/calibration.py](src/calibration.py) 
file. Calibration can be performed as follows, as being done in the IPython notebook
[exploration.ipynb](exploration.ipynb).

```python
from src import calibration
camera = calibration.Camera()
camera.calibrate()
camera.undistort(img)
```

Calibration starts with preparation of "object points", which are the (x, y, z) coordinates of the chessboard corners 
in the real world. We can assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points 
are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` are 
appended with a copy of it every time all chessboard corners are detected in a test image. Similarly, `imgpoints` are
appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard 
detection.  

The lists of `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using
`cv2.calibrateCamera` function from OpenCV. Then, distortion correction can be applied to any image using 
`cv2.undistort` function. Here is an example of how distortion correction is applied to one of the chess board images.
 
![Original chess board][original-chess-board] ![Undistorted chess board][undistorted-chess-board]

Notice how the top central boxes in the chess board appear to have been "straightened" after this
distortion correction.


### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
