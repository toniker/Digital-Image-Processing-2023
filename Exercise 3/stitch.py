import random
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
        number_of_descriptors = (rho_M - rho_m) // rho_step
        return np.array([np.inf] * number_of_descriptors)

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
        number_of_descriptors = (rho_M - rho_m) // rho_step
        return np.array([np.inf] * number_of_descriptors)

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


def is_corner(p, k, r_threshold, Ix, Iy):
    def w(x, y):
        sigma = 1
        return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    p1, p2 = p
    if is_out_of_bounds(Ix, (p1, p2), 5):
        return False

    M = np.zeros((2, 2))
    for u1 in range(-2, 3):
        for u2 in range(-2, 3):
            Ixy = Ix[p1 + u1, p2 + u2] * Iy[p1 + u1, p2 + u2]
            A = np.array([[Ix[p1 + u1, p2 + u2] ** 2, Ixy],
                          [Ixy, Iy[p1 + u1, p2 + u2] ** 2]])
            M += w(u1, u2) * A

    r = np.linalg.det(M) - k * (np.trace(M) ** 2)

    return r > r_threshold


def my_detect_harris_features(I):
    k = 0.04
    r_threshold = 0.02

    I_with_corners = I.copy()
    I_with_corners = I_with_corners * 255
    I_with_corners = I_with_corners.astype(np.uint8)
    I_with_corners = cv2.cvtColor(I_with_corners, cv2.COLOR_GRAY2RGB)

    Ix = np.gradient(I, axis=0)
    Iy = np.gradient(I, axis=1)
    corners = []
    height, width = I.shape
    for y in range(height):
        for x in range(width):
            if is_corner((y, x), k, r_threshold, Ix, Iy):
                corners.append((y, x))
                cv2.circle(I_with_corners, (x, y), 1, (255, 0, 0), 1)

    cv2.imwrite(f"corners.png", I_with_corners)
    return corners


def descriptor_matching(points_1, points_2, percentage_threshold):
    distances = np.empty((len(points_1), len(points_2)))
    matching_points = []

    for index_1, point_1 in enumerate(points_1):
        for index_2, point_2 in enumerate(points_2):
            point_1 = np.array(point_1)
            point_2 = np.array(point_2)
            distances[index_1, index_2] = np.linalg.norm(point_1 - point_2)

    np.save("distances.npy", distances)
    distances = np.load("distances.npy")

    for index_1, point_1 in enumerate(points_1):
        minimum_value_index = np.argmin(distances[index_1])
        matching_points.append((index_1, minimum_value_index))

    _matching_points = [x for x in matching_points if x[1] != 0]
    matching_points = _matching_points

    return matching_points


def my_RANSAC(matching_points, r, N):
    inlier_matching_points = []
    outlier_matching_points = []
    theta, d = None, None

    for i in range(N):
        random_points = random.sample(matching_points, 2)
        point_1, point_2 = random_points[0], random_points[1]
        point_1 = point_1[0]
        point_2 = point_2[0]

        y1, x1 = point_1
        y2, x2 = point_2

        theta = np.arctan((y2 - y1) / (x2 - x1))
        d = y1 - x1 * (y2 - y1) / (x2 - x1)

        inlier_matching_points = []
        outlier_matching_points = []
        for point in matching_points:
            y, x = point[0]
            distance = np.abs(y - x * np.tan(theta) - d)
            if distance < r:
                inlier_matching_points.append(point)
            else:
                outlier_matching_points.append(point)

        if len(inlier_matching_points) > len(outlier_matching_points):
            break

    H = {'theta': theta, 'd': d}
    return H, inlier_matching_points, outlier_matching_points


def my_stitch(im1, im2):
    grey_im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    grey_im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    grey_im1_float = grey_im1.astype(np.float32) / np.max(grey_im1)
    grey_im2_float = grey_im2.astype(np.float32) / np.max(grey_im2)

    im1_data = {'image': im1, 'grey_image': grey_im1, 'corners': my_detect_harris_features(grey_im1_float)}
    im2_data = {'image': im2, 'grey_image': grey_im2, 'corners': my_detect_harris_features(grey_im2_float)}

    rho_m = 5
    rho_M = 20
    rho_step = 1
    N = 8

    im1_data['descriptors'] = []
    for corner in im1_data['corners']:
        im1_data['descriptors'].append(my_local_descriptor(grey_im1, corner, rho_m, rho_M, rho_step, N))

    im2_data['descriptors'] = []
    for corner in im2_data['corners']:
        im2_data['descriptors'].append(my_local_descriptor(grey_im2, corner, rho_m, rho_M, rho_step, N))

    data = {'im1': im1_data, 'im2': im2_data}
    np.save('data.npy', data)
    # data = np.load('data.npy', allow_pickle=True).tolist()
    # im1_data = data['im1']
    # im2_data = data['im2']
    #
    # matching_points = descriptor_matching(im1_data['descriptors'], im2_data['descriptors'], percentage_threshold=0.8)
    #
    # comb = cv2.imread("combined.png")
    # np.random.shuffle(matching_points)
    # for i in range(len(matching_points) // 20):
    #     matching_point = matching_points[i]
    #     index_1, index_2 = matching_point
    #     (y1, x1) = im1_data['corners'][index_1]
    #     (y2, x2) = im2_data['corners'][index_2]
    #     x2 += 1360
    #     cv2.line(comb, (x1, y1), (x2, y2), (0, 255, 0))
    # cv2.imwrite("combined_lines.jpg", comb)
    #
    # r = 5
    # N = 100
    # H, inlier_matching_points, outlier_matching_points = my_RANSAC(matching_points, r, N)
    #
    # print("theta: ", H['theta'], "d: ", H['d'])
    stitched = 0
    return stitched


if __name__ == "__main__":
    start_time = time.time()
    im1 = cv2.imread("im1.png")
    im2 = cv2.imread("im2.png")

    rho_m = 5
    rho_M = 20
    rho_step = 1
    N = 8

    grey_im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    deliverable_1 = my_local_descriptor(grey_im1, (100, 100), rho_m, rho_M, rho_step, N)
    deliverable_2_1 = my_local_descriptor(grey_im1, (200, 200), rho_m, rho_M, rho_step, N)
    deliverable_2_2 = my_local_descriptor(grey_im1, (202, 202), rho_m, rho_M, rho_step, N)

    deliverable_1_upgrade = my_local_descriptor_upgrade(grey_im1, (100, 100), rho_m, rho_M, rho_step, N)
    deliverable_2_1_upgrade = my_local_descriptor_upgrade(grey_im1, (200, 200), rho_m, rho_M, rho_step, N)
    deliverable_2_2_upgrade = my_local_descriptor_upgrade(grey_im1, (202, 202), rho_m, rho_M, rho_step, N)

    stitched_city = my_stitch(im1, im2)
    # cv2.imwrite("stitched_city.jpg", stitched_city)

    # im_forest1 = cv2.imread("imforest1.png")
    # im_forest2 = cv2.imread("imforest2.png")
    # stitched_forest = my_stitch(im_forest1, im_forest2)
    print(f'Execution time: {time.time() - start_time}')
