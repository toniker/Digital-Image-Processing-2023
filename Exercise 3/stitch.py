import time

import cv2
import numpy as np


def is_out_of_bounds(I, p, rho_M):
    """
    Check if the point p plus the radius rhoM is out of bounds of the image I.
    :param I: the image
    :param p: the point
    :param rho_M: the radius
    :return: True if the point is out of bounds, False otherwise
    """
    x, y = p
    return x + rho_M > I.shape[0] or x - rho_M < 0 or y + rho_M > I.shape[1] or y - rho_M < 0


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


def my_local_descriptor(I, p, rho_m, rho_M, rho_step, N):
    if is_out_of_bounds(I, p, rho_M):
        return np.array([])

    descriptor = []
    for radius in range(rho_m, rho_M, rho_step):
        points = []
        for angle in range(0, 360, N):
            xy_offset = pol2cart(radius, angle)
            point = tuple(int(sum(x)) for x in zip(p, xy_offset))
            points.append(I[point])

        points = np.array(points)
        descriptor.append(np.mean(points))
    return descriptor


def my_local_descriptor_upgrade(I, p, rho_m, rho_M, rho_step, N):
    if is_out_of_bounds(I, p, rho_M):
        return np.array([])

    descriptor = []
    for radius in range(rho_m, rho_M, rho_step):
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


def is_corner(I, p, k, r_threshold):
    def w(x, y):
        sigma = 1
        return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    x1, x2 = 10, 10

    cum_sum = 0.0
    p1, p2 = p
    for u1 in range(-5, 6):
        for u2 in range(-5, 6):
            cum_sum += w(u1, u2) * (I[u1 + p1 + x1, u2 + p2 + x2] - I[u1 + p1, u2 + p2]) ** 2

    return cum_sum > r_threshold


def my_detect_harris_features(I):
    k = 0.08
    r_threshold = 2
    corners = []
    height, width = I.shape
    I = cv2.copyMakeBorder(I, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
    I_with_corners = I.copy()
    I_with_corners = cv2.cvtColor(I_with_corners, cv2.COLOR_GRAY2RGB)
    I_with_corners = I_with_corners * 255
    I_with_corners = I_with_corners.astype(np.uint8)

    for y in range(height):
        for x in range(width):
            if is_corner(I, (y, x), k, r_threshold):
                corners.append((y, x))
                cv2.circle(I_with_corners, (x, y), 1, (255, 0, 0), 1)

    cv2.imwrite(f"corners_k_{k}_r_{r_threshold}.png", I_with_corners)
    return corners


def euclidean_distance(point_1, point_2):
    difference = [point_2[0] - point_2[0], point_2[1] - point_1[1]]
    return np.sqrt(np.sum([x ** 2 for x in difference]))


def descriptor_matching(points_1, points_2, percentage_threshold):
    distances = np.array((len(points_1), len(points_2)))
    for index_1, point_1 in enumerate(points_1):
        for index_2, point_2 in enumerate(points_2):
            distance = euclidean_distance(point_1, point_2)
            distances[index_1, index_2] = distance

    for index in distances:
        pass
    return matching_points


def my_RANSAC(matching_points, r, N):
    H = {'theta': theta, 'd': d}
    return (H, inlier_matching_points, outlier_matching_points)


def my_stitch(im1, im2):
    im1_harris = my_detect_harris_features(im1)
    im2_harris = my_detect_harris_features(im2)

    return stitched


if __name__ == "__main__":
    start_time = time.time()
    start_time = time.time()
    im1 = cv2.imread("im1.png")
    im2 = cv2.imread("im2.png")

    grey_im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    deliverable_1 = my_local_descriptor(grey_im1, (100, 100), 5, 20, 1, 8)
    deliverable_2_1 = my_local_descriptor(grey_im1, (200, 200), 5, 20, 1, 8)
    deliverable_2_2 = my_local_descriptor(grey_im1, (202, 202), 5, 20, 1, 8)

    deliverable_1_upgrade = my_local_descriptor_upgrade(grey_im1, (100, 100), 5, 20, 1, 8)
    deliverable_2_1_upgrade = my_local_descriptor_upgrade(grey_im1, (200, 200), 5, 20, 1, 8)
    deliverable_2_2_upgrade = my_local_descriptor_upgrade(grey_im1, (202, 202), 5, 20, 1, 8)

    grey_im1_float = grey_im1.astype(np.float32) / np.max(grey_im1)
    corners = my_detect_harris_features(grey_im1_float)

    # stitched_city = my_stitch(im1, im2)

    # im_forest1 = cv2.imread("imforest1.png")
    # im_forest2 = cv2.imread("imforest2.png")
    # stitched_forest = my_stitch(im_forest1, im_forest2)
    print(f'Execution time: {time.time() - start_time}')
