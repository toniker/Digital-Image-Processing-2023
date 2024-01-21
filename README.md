# Digital Image Processing

This repository contains the source code for the projects of the class Digital Image Processing, taken during my study at the school of Electrical and Computer Engineering at AUTh during the spring semester of 2023.

Each folder contains a self-contained exercise that serves a specific purpose over given input images.

# Exercise 1

A RAW Image converter is implemented in MATLAB, using custom debayer functions. Images using different debayering techniques are shown below.

| Built-in MATLAB function | Custom Function |
|---------------- | --------------- |
| ![builtInDemosaic](https://github.com/toniker/digital-image-processing-2023/assets/39350193/9f855ef3-5a2c-4827-bd08-bccb3d7beff9)| ![linear_rggb_rgb](https://github.com/toniker/digital-image-processing-2023/assets/39350193/e61cb545-4cd0-4f06-9831-be0df31add17) |

# Exercise 2

An OCR implementation is developed using Python. First, the rotation angle of the image is detected using the 2D DFT of the image. We can rotate the image to the appropriate angle for valid text recognition. 

The next part of the process includes the detection, representation and comparison of letter contours. This process includes thresholding and binary operations on the brightness of letter images. The comparison of contours happens using the DFT of the contour representations.

Finally, I developed an algorithm to recognize lines, words and letters in a given image. This is done by calculating the differential of the image brightness. Using the projection of this differential we can use thresholding to detect lines in the vertical axis, then words and letters in the horizontal axis.

| Original Text | Recognized words replaced with bounding boxes |
|---------------- | --------------- |
| ![text1_v3](https://github.com/toniker/digital-image-processing-2023/assets/39350193/318ce30e-c346-4c02-baef-bb6cac891ff4) | ![blank](https://github.com/toniker/digital-image-processing-2023/assets/39350193/cfbd491e-64e0-4474-9052-6eb1395228e3) |

# Exercise 3

This exercise attempts to stitch two satellite photos of a city. This is done by detecting features (corners) in each image, describing these features using a local descriptor, matching these descriptors across the two images, and then using RANSAC to robustly estimate the transformation (rotation and translation) between the two images. The second image is then transformed according to this estimated transformation and overlaid onto the first image to produce the final stitched image.

