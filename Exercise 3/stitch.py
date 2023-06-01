import numpy as np
import cv2


def my_local_descriptor(I, p, rhom, rhoM, rhostep, N):
    return descriptor


def my_local_descriptor_upgrade(I, p, rhom, rhoM, rhostep, N):
    return descriptor


def my_stitch(im1, im2):
    return stitched


def isCorner(I, p, k, r_threshold):
    return bool


def myDetectHarrisFeatures(I):
    return corners


def descriptorMatching(points1, points2, percentage_threshold):
    return matching_points


def myRANSAC(matching_points, r, N):
    return (H, inlier_matching_points, outlier_matching_points)


if __name__ == "__main__":
    im1 = cv2.imread("im1.png")
    im2 = cv2.imread("im2.png")

    stitched_city = my_stitch(im1, im2)

    imforest1 = cv2.imread("imforest1.png")
    imforest2 = cv2.imread("imforest2.png")

    stitched_forest = my_stitch(imforest1, imforest2)
    breakpoint()
