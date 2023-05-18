import time

import cv2
import numpy as np


def find_rotation_angle_hough(image):
    """
    Uses a Hough transform to find the rotation angle of the image.
    Using the DFT of the image, we find the lines corresponding to the text and use the slope of those lines to find the
    angle of the text.
    :param image: The input image
    :return: The estimated rotation angle
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.blur(image, (15, 15))
    f = np.fft.fft2(blurred_image)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    magnitude_spectrum = magnitude_spectrum - np.mean(magnitude_spectrum)
    # Apply thresholding
    magnitude_spectrum[magnitude_spectrum < 130] = 0
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
    magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("magnitude_spectrum.jpg", magnitude_spectrum)

    # Apply edge detection
    edges = cv2.Canny(magnitude_spectrum, 40, 150)

    # Apply Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=70, maxLineGap=20)

    if lines is None:
        return None

    angles = []
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        predicted_angle = np.degrees(np.arctan(slope))
        if not -35 < predicted_angle < 35:
            continue
        angles.append(predicted_angle)
        cv2.line(magnitude_spectrum, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("hough_lines_on_magnitude_spectrum.jpg", magnitude_spectrum)

    return -round(float(np.mean(angles)), 2)


def find_rotation_angle(image):
    """
    Uses the methodology described in the homework to find the rotation angle of the image.
    Using the DFT of the image, we find the angle of the maximum frequency, to the center of the image. With that
    initial estimate, we rotate the image and use the vertical projection until we find the angle that gives us the most
    shifts between bright and dark areas.
    :param image: The input image
    :return: The estimated angle
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.blur(image, (8, 8))

    f = np.fft.fft2(blurred_image)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    center = tuple(i // 2 for i in magnitude_spectrum.shape)
    magnitude_spectrum[center] = 0

    max_frequency = np.argmax(magnitude_spectrum)
    max_frequency_position = np.unravel_index(max_frequency, magnitude_spectrum.shape)

    initial_estimation = np.rad2deg(
        np.arctan2(center[1] - max_frequency_position[1], center[0] - max_frequency_position[0]))
    initial_estimation = int(initial_estimation)
    initial_estimation %= 90

    best_estimate = initial_estimation, 0
    for possible_angle in range(initial_estimation - 20, initial_estimation + 20):
        test_image = fast_rotate_image(image, -possible_angle)
        test_image = cv2.blur(test_image, (11, 11))
        test_image[test_image < 240] = 0
        test_image[test_image >= 240] = 255
        vertical_projection = np.sum(test_image, axis=0)
        diffs = [abs(vertical_projection[j + 1] - vertical_projection[j]) for j in range(len(vertical_projection) - 1)]
        avg_diff = np.mean(diffs)
        if avg_diff > best_estimate[1]:
            best_estimate = (possible_angle, avg_diff)

    return best_estimate[0]


def rotate_image(input_image, angle):
    """
    Rotates the image by the given angle.
    :param input_image: The image to be rotated.
    :param angle: The angle to rotate the image.
    :return: The rotated image.
    """
    radians = np.radians(angle)
    input_height, input_width, number_of_colors = np.shape(input_image)
    input_center = (input_height // 2, input_width // 2)

    output_height = abs(input_height * np.cos(radians) + input_width * np.sin(radians))
    output_width = abs(input_height * np.sin(radians) + input_width * np.cos(radians))
    output_height, output_width = int(np.round(output_height)), int(np.round(output_width))
    output_center = (output_height // 2, output_width // 2)

    rotated_image = np.zeros((output_height, output_width, number_of_colors), dtype=np.uint8)

    for i in range(output_height):
        for j in range(output_width):
            x = (i - output_center[0]) * np.cos(radians) + (j - output_center[1]) * np.sin(radians) + input_center[0]
            y = -(i - output_center[0]) * np.sin(radians) + (j - output_center[1]) * np.cos(radians) + input_center[1]
            x = int(np.round(x))
            y = int(np.round(y))
            if 0 <= x < input_height and 0 <= y < input_width:
                rotated_image[i, j] = input_image[x, y]

    return rotated_image


def fast_rotate_image(image, angle):
    """
    Uses OpenCV to rotate the image.
    This function was created since it is orders of magnitude faster than the rotate_image function.
    :param image: The image to be rotated.
    :param angle: The angle to rotate the image.
    :return: The rotated image.
    """
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # Calculate new image dimensions
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
    rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

    # Perform the actual rotation and pad the unused area with black
    result = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return result


if __name__ == "__main__":
    start_time = time.time()
    img = cv2.imread("text1.png")

    rot_image = fast_rotate_image(img, 0)

    angle = find_rotation_angle_hough(rot_image)
    print(f"Estimated angle: {angle} degrees")

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Rotation finished in {execution_time} seconds")
