import numpy as np
import cv2


def get_line_indices(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    _, binarizedImage = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    vertical_projection = np.sum(binarizedImage, axis=1)
    vertical_projection[vertical_projection < np.max(vertical_projection)] = 0
    vertical_projection[vertical_projection > 0] = 1

    lines = []
    start_index = None
    for i, val in enumerate(vertical_projection):
        if val == 0 and start_index is None:
            start_index = i
        elif val == 1 and start_index is not None:
            lines.append((start_index, i - 1))
            start_index = None

    if start_index is not None:
        lines.append((start_index, len(vertical_projection) - 1))

    return lines


if __name__ == '__main__':
    img = cv2.imread('text1.png')
    lines = get_line_indices(img)