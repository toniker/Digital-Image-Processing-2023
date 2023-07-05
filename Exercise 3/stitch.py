import multiprocessing
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
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / N):
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
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / N):
            xy_offset = pol2cart(radius, angle)
            point = tuple(int(sum(x)) for x in zip(p, xy_offset))
            points.append(I[point])

            xy_offset = [v // 2 for v in xy_offset]
            point = tuple(int(sum(x)) for x in zip(p, xy_offset))
            points.append(I[point] ** 2)

        points = np.array(points)
        descriptor.append(np.mean(points))
    return descriptor


def is_corner(p, k, r_threshold, Ixy):
    def w(x, y):
        sigma = 1
        return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    p1, p2 = p
    if is_out_of_bounds(Ixy, (p1, p2), 2):
        return False

    cumulative_sum = 0
    for u1 in range(-2, 2):
        for u2 in range(-2, 2):
            cumulative_sum += w(u1, u2) * (Ixy[u1 + p1, u2 + p2]) ** 2

    return cumulative_sum > r_threshold


def process_corner(y, x, k, r_threshold, Ixy):
    if is_corner((y, x), k, r_threshold, Ixy):
        return y, x


def my_detect_harris_features(I):
    k = 0.04
    r_threshold = 0.3

    Ix, Iy = np.gradient(I)
    Ixy = Ix + Iy
    Ixy = np.clip(Ixy, 0, 255)
    height, width = I.shape
    pool = multiprocessing.Pool()

    # Create a list of coordinates for all pixels
    coordinates = [(y, x) for y in range(height) for x in range(width)]

    # Map the processing function to the list of coordinates using multiple processes
    results = pool.starmap(process_corner, [(y, x, k, r_threshold, Ixy) for (y, x) in coordinates])

    # Filter out None values from the results and extract the corner coordinates
    corners = [corner for corner in results if corner is not None]

    pool.close()
    pool.join()
    return corners


def descriptor_matching(points_1, points_2, percentage_threshold):
    distances = np.empty((len(points_1), len(points_2)))

    for index_1, point_1 in enumerate(points_1):
        for index_2, point_2 in enumerate(points_2):
            point_1 = np.array(point_1)
            point_2 = np.array(point_2)
            distances[index_1, index_2] = np.linalg.norm(point_1 - point_2)

    np.save("distances.npy", distances)
    distances = np.load("distances.npy")

    matching_points = []
    for index_1 in range(len(distances)):
        sorted = np.argsort(distances[index_1])
        accepted = sorted[:int(len(sorted) * percentage_threshold)]
        filtered = accepted[np.isfinite(distances[index_1, accepted])]
        for index_2 in filtered:
            matching_points.append((index_1, index_2))

    return matching_points


def my_RANSAC(matching_points, r, N, im1_data, im2_data):
    def transform_points(points, theta, d):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        homogeneous_matrix = np.column_stack((rotation_matrix, d))
        homogeneous_matrix = np.vstack((homogeneous_matrix, [0, 0, 1]))
        homogeneous_points = np.column_stack((points, np.ones(len(points))))

        transformed_points = np.dot(homogeneous_points, homogeneous_matrix.T)
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
        return transformed_points

    inlier_matching_points = []
    outlier_matching_points = []
    H = {'theta': 0, 'd': 0}
    best_distance = np.inf
    distances = []

    for i in range(N):
        pairs_of_matching_points = random.sample(matching_points, 2)
        pair_1, pair_2 = pairs_of_matching_points
        im1_y1, im1_x1 = im1_data['corners'][pair_1[0]]
        im2_y1, im2_x1 = im2_data['corners'][pair_1[1]]
        im1_y2, im1_x2 = im1_data['corners'][pair_2[0]]
        im2_y2, im2_x2 = im2_data['corners'][pair_2[1]]

        dx = (im2_x1 - im1_x1) + (im2_x2 - im1_x2) // 2
        dy = (im2_y1 - im1_y1) + (im2_y2 - im1_y2) // 2

        d = [dx, dy]
        im1_theta = np.arctan2(im1_y1, im1_x1)
        im2_theta = np.arctan2(im2_y1, im2_x1)
        theta = im1_theta - im2_theta

        transformed_im1_points = np.array(im1_data['corners'])
        transformed_im2_points = transform_points(im2_data['corners'], theta, d)

        number_of_points = 100000
        random_selection = random.sample(matching_points, number_of_points)
        im2_points = transformed_im2_points[np.array([pair[1] for pair in random_selection])]
        im1_points = transformed_im1_points[np.array([pair[0] for pair in random_selection])]

        distance = np.linalg.norm(im2_points - im1_points) // number_of_points
        distances.append(distance)
        if distance < best_distance:
            best_distance = distance
            H['theta'] = theta
            H['d'] = d

        if distance < r:
            inlier_matching_points.append(pairs_of_matching_points)
        else:
            outlier_matching_points.append(pairs_of_matching_points)

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
    data = np.load('data.npy', allow_pickle=True).tolist()
    im1_data = data['im1']
    im2_data = data['im2']

    grey_im1 = cv2.cvtColor(grey_im1, cv2.COLOR_GRAY2RGB)
    for (y, x) in im1_data['corners']:
        cv2.circle(grey_im1, (x, y), 1, (0, 255, 0), 1)

    cv2.imwrite(f"im1_corners.png", grey_im1)

    grey_im2 = cv2.cvtColor(grey_im2, cv2.COLOR_GRAY2RGB)
    for (y, x) in im2_data['corners']:
        cv2.circle(grey_im2, (x, y), 1, (0, 255, 0), 1)

    cv2.imwrite(f"im2_corners.png", grey_im2)
    # matching_points = descriptor_matching(im1_data['descriptors'], im2_data['descriptors'], percentage_threshold=0.05)
    #
    # np.save('matching_points.npy', matching_points)
    matching_points = np.load('matching_points.npy', allow_pickle=True).tolist()

    r = 3
    N = 100
    H, inlier_matching_points, outlier_matching_points = my_RANSAC(matching_points, r, N, im1_data, im2_data)

    im1_inliers = im1
    im2_inliers = im2
    for inlier in inlier_matching_points:
        pair_1, pair_2 = inlier
        im1_y1, im1_x1 = im1_data['corners'][pair_1[0]]
        im2_y1, im2_x1 = im2_data['corners'][pair_1[1]]
        im1_y2, im1_x2 = im1_data['corners'][pair_2[0]]
        im2_y2, im2_x2 = im2_data['corners'][pair_2[1]]

        cv2.circle(im1_inliers, (im1_x1, im1_y1), 3, (0, 0, 255), 3)
        cv2.circle(im1_inliers, (im1_x2, im1_y2), 3, (0, 0, 255), 3)
        cv2.circle(im2_inliers, (im2_x1, im2_y1), 3, (0, 0, 255), 3)
        cv2.circle(im2_inliers, (im2_x2, im2_y2), 3, (0, 0, 255), 3)

    cv2.imwrite(f"im1_inliers.png", im1_inliers)
    cv2.imwrite(f"im2_inliers.png", im2_inliers)
    print("theta: ", H['theta'], "d: ", H['d'])
    dx, dy = H['d']
    theta = np.rad2deg(H['theta'])

    im2_center = (im2.shape[1] // 2 + dx, im2.shape[0] // 2 + dy)
    new_width = int((im2.shape[1] + abs(dx)) * np.cos(theta) + (im2.shape[0] + abs(dy)) * np.sin(theta))
    new_height = int((im2.shape[1] + abs(dx)) * np.sin(theta) + (im2.shape[0] + abs(dy)) * np.cos(theta))
    rotate_matrix = cv2.getRotationMatrix2D(center=im2_center, angle=theta, scale=1)
    transformed_im2 = cv2.warpAffine(src=im2, M=rotate_matrix, dsize=(new_width, new_height))

    h, w = transformed_im2.shape[:2]

    im2_alpha = np.dstack((transformed_im2, np.zeros((h, w), dtype=np.uint8) + 255))
    mBlack = (im2_alpha[:, :, 0:3] == [0, 0, 0]).all(2)
    im2_alpha[mBlack] = (0, 0, 0, 0)

    combined_width = max(im1.shape[1], transformed_im2.shape[1] + dx)
    combined_height = max(im1.shape[0], transformed_im2.shape[0] + dy)
    stitched = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    stitched[0:im1.shape[0], 0:im1.shape[1]] = im1
    stitched[dy:transformed_im2.shape[0] + dy, dx:transformed_im2.shape[1] + dx] = transformed_im2
    cv2.imwrite("stitched.png", stitched)
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
