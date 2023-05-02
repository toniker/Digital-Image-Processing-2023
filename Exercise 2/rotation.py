import time

import cv2 as cv
import numpy as np


def find_rotation_angle(img) -> float:
    """
    This function finds the rotation angle of the image.
    :param img:
    :return: the angle as a float
    """
    angle = 0

    return angle


def rotate_image(input_image, angle):
    radians = np.radians(angle)
    input_height, input_width, number_of_colors = np.shape(input_image)
    input_center = (input_height // 2, input_width // 2)

    output_height = input_height * np.cos(radians) + input_width * np.sin(radians)
    output_width = input_height * np.sin(radians) + input_width * np.cos(radians)
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


if __name__ == "__main__":
    start_time = time.time()
    image = cv.imread("image.jpg")
    cv.imwrite("rotated.jpg", rotate_image(image, 60))
    # Measure the execution time
    execution_time = round(time.time() - start_time, 3)
    print(f"Rotation finished in {execution_time} seconds")
