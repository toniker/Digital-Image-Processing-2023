import cv2
import numpy as np


def is_out_of_bounds(I, p, rhoM):
    """
    Check if the point p plus the radius rhoM is out of bounds of the image I.
    :param I: the image
    :param p: the point
    :param rhoM: the radius
    :return: True if the point is out of bounds, False otherwise
    """
    x, y = p
    if x + rhoM > I.shape[0] or x - rhoM < 0 or y + rhoM > I.shape[1] or y - rhoM < 0:
        return True
    return False


def pol2cart(rho, phi):
    """
    Convert polar coordinates to cartesian coordinates
    :param rho: the radius
    :param phi: the angle in radians
    :return: the x and y coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def my_local_descriptor(I, p, rhom, rhoM, rhostep, N):
    if is_out_of_bounds(I, p, rhoM):
        return np.array([])

    descriptor = []
    for radius in range(rhom, rhoM, rhostep):
        points = []
        for angle in range(0, 360, N):
            xy_offset = pol2cart(radius, angle)
            point = tuple(int(sum(x)) for x in zip(p, xy_offset))
            points.append(I[point])

        points = np.array(points)
        descriptor.append(np.mean(points))
    return descriptor


def my_local_descriptor_upgrade(I, p, rhom, rhoM, rhostep, N):
    if is_out_of_bounds(I, p, rhoM):
        return np.array([])

    descriptor = []
    for radius in range(rhom, rhoM, rhostep):
        points = []
        for angle in range(0, 360, N):
            xy_offset = pol2cart(radius, angle)
            point = tuple(int(sum(x)) for x in zip(p, xy_offset))
            points.append(I[point])

            xy_offset = [v // 2 for v in xy_offset]
            point = tuple(int(sum(x)) for x in zip(p, xy_offset))
            points.append(I[point] ** 2)

        points = np.array(points)
        descriptor.append(np.mean(points))
    return descriptor


def my_stitch(im1, im2):
    return stitched


def is_corner(I, p, k, r_threshold):
    return bool


def my_detect_harris_features(I):
    return corners


def descriptor_matching(points1, points2, percentage_threshold):
    return matching_points


def my_RANSAC(matching_points, r, N):
    return (H, inlier_matching_points, outlier_matching_points)


if __name__ == "__main__":
    im1 = cv2.imread("im1.png")
    im2 = cv2.imread("im2.png")

    grey_im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    deliverable_1 = my_local_descriptor(grey_im1, (100, 100), 5, 20, 1, 8)
    deliverable_2_1 = my_local_descriptor(grey_im1, (200, 200), 5, 20, 1, 8)
    deliverable_2_2 = my_local_descriptor(grey_im1, (202, 202), 5, 20, 1, 8)

    deliverable_1_upgrade = my_local_descriptor_upgrade(grey_im1, (100, 100), 5, 20, 1, 8)
    deliverable_2_1_upgrade = my_local_descriptor_upgrade(grey_im1, (200, 200), 5, 20, 1, 8)
    deliverable_2_2_upgrade = my_local_descriptor_upgrade(grey_im1, (202, 202), 5, 20, 1, 8)
    stitched_city = my_stitch(im1, im2)

    imforest1 = cv2.imread("imforest1.png")
    imforest2 = cv2.imread("imforest2.png")

    stitched_forest = my_stitch(imforest1, imforest2)
    breakpoint()
