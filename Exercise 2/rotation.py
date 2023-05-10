import time

import cv2
import numpy as np


def find_rotation_angle(image: np.ndarray) -> float:
    """
    This function finds the rotation angle of the image.
    :param image: The input image
    :return: the angle as a float
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.blur(image, (15, 15))
    f = np.fft.fft2(blurred_image)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    magnitude_spectrum = magnitude_spectrum - np.mean(magnitude_spectrum)
    # Apply thresholding
    magnitude_spectrum[magnitude_spectrum < 150] = 0
    cv2.imwrite("magnitude_spectrum.jpg", magnitude_spectrum)

    # Apply edge detection
    edges = cv2.Canny(image, 50, 150)

    # Apply Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=10)

    angles = []
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        angle = np.degrees(np.arctan(slope))
        angles.append(angle)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("lines.jpg", image)

    return float(np.mean(angles))


def rotate_image(input_image, angle):
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
    image = cv2.imread("text1.png")

    rotated_image = fast_rotate_image(image, 20)
    # cv.imwrite("rotated.jpg", rotated_image)

    angle = find_rotation_angle(rotated_image)
    print(f"Angle: {angle}")

    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Rotation finished in {execution_time} seconds")
