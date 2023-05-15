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


def get_word_indices(image, line_indices):
    indices = []
    blank_image = np.ones_like(image) * 255
    for line in line_indices:
        line_image = image[line[0]:line[1], :]
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        line_image = cv2.GaussianBlur(line_image, (55, 55), 0)

        _, binarizedImage = cv2.threshold(line_image, 200, 255, cv2.THRESH_BINARY)

        horizontal_projection = np.sum(binarizedImage, axis=0)
        horizontal_projection[horizontal_projection < np.max(horizontal_projection)] = 0
        horizontal_projection[horizontal_projection > 0] = 1

        words = []
        start_index = None
        for i, val in enumerate(horizontal_projection):
            if val == 0 and start_index is None:
                start_index = i
            elif val == 1 and start_index is not None:
                words.append((start_index, i - 1))
                start_index = None

        if start_index is not None:
            words.append((start_index, len(horizontal_projection) - 1))

        for index in indices:
            for word in words:
                blank_image[line[0]:line[1], word[0]:word[1], :] = 0
        indices.append(words)

    return indices


if __name__ == '__main__':
    img = cv2.imread('text1_v2.png')
    lines = get_line_indices(img)
    words = get_word_indices(img, lines)

    for i, line in enumerate(words):
        for j, word in enumerate(line):
            cv2.imwrite('lines/text1_line{}_word{}.png'.format(i, j), img[lines[i][0]:lines[i][1], word[0]:word[1]])
    print('Done')
