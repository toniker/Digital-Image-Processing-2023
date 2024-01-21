# Digital Image Processing

This repository contains the source code for the projects of the class Digital Image Processing, taken during my study at the school of Electrical and Computer Engineering at AUTh during the spring semester of 2023.

Each folder contains a self-contained exercise that serves a specific purpose over given input images.

# Exercise 1

A RAW Image converter is implemented in MATLAB, using custom debayer functions. Images using different debayering techniques are shown below.

# Add images

# Exercise 2

An OCR implementation is developed using Python. First, the rotation angle of the image is detected using the 2D DFT of the image. We can rotate the image to the appropriate angle for valid text recognition. 

The next part of the process includes the detection, representation and comparison of letter contours. This process includes thresholding and binary operations on the brightness of letter images. The comparison of contours happens using the DFT of the contour representations.

Finally, I developed an algorithm to recognize lines, words and letters in a given image. This is done by calculating the differential of the image brightness. Using the projection of this differential we can use thresholding to detect lines in the vertical axis, then words and letters in the horizontal axis.

# Add images 

# Exercise 3

This exercise attempts to stitch two satellite photos of a city. This is done by detecting features (corners) in each image, describing these features using a local descriptor, matching these descriptors across the two images, and then using RANSAC to robustly estimate the transformation (rotation and translation) between the two images. The second image is then transformed according to this estimated transformation and overlaid onto the first image to produce the final stitched image.

