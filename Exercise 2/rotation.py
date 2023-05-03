import time

import cv2
import numpy as np


def find_rotation_angle(image) -> float:
    """
    This function finds the rotation angle of the image.
    :param img: The input image
    :return: the angle as a float
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_frequencies = np.array([])

    for possible_angle in range(0, 360):
        rotated_image = fast_rotate_image(image, possible_angle)
        blurred_image = cv2.blur(rotated_image, (15, 15))
        f = np.fft.fft2(blurred_image)
        f_shift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))

        # Find the maximum value of the middle row
        vertical_axis = magnitude_spectrum.shape[0] // 2
        max_value_index = np.argmax(magnitude_spectrum[:, vertical_axis])

        # Convert index to frequency value
        n = magnitude_spectrum.shape[0]
        # The frequency values will be in the range of -0.5 to 0.5 cycles per pixel, as explained in the previous answer
        max_frequency = max_value_index / n
        max_frequencies = np.append(max_frequencies, max_frequency)

    # cv2.imwrite("magnitude.jpg", magnitude_spectrum)
    angle = np.argmax(max_frequencies)
    print(f"Max frequency: {max(max_frequencies)} at angle {angle}")

    return angle


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
    # cv2.imwrite("fast_rotated.jpg", fast_rotate_image(image, 144))
    # rotated_image = rotate_image(image, 30)
    # cv.imwrite("rotated.jpg", rotated_image)
    # rotated_image = cv.imread("rotated.jpg")
    angle = find_rotation_angle(image)
    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Rotation finished in {execution_time} seconds")
