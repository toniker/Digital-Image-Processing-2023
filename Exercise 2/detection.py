import numpy as np
import cv2
from contour import get_contour, compare_contours


class Letter:
    def __init__(self, x1, x2, y1, y2, contour=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.coordinates = ((x1, y1), (x2, y2))
        self.contour = contour


class Word:
    def __init__(self, x1, x2, y1, y2, letters=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.coordinates = ((x1, y1), (x2, y2))
        self.letters = letters


class Line:
    def __init__(self, y1, y2, words=None):
        self.y1 = y1
        self.y2 = y2
        self.coordinates = (y1, y2)
        self.words = words


def get_line_indices(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (25, 25), 0)

    _, binarizedImage = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    vertical_projection = np.sum(binarizedImage, axis=1)
    vertical_projection[vertical_projection < np.max(vertical_projection)] = 0
    vertical_projection[vertical_projection > 0] = 255

    lines = []
    start_index = None
    for i, val in enumerate(vertical_projection):
        if val == 0 and start_index is None:
            start_index = i
        elif val == 255 and start_index is not None:
            lines.append(Line(start_index, i - 1))
            start_index = None

    if start_index is not None:
        lines.append(Line(start_index, len(vertical_projection) - 1))

    return lines


def get_words(image, lines):
    indices = []

    for line in lines:
        line_image = image[line.y1:line.y2, :]
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        line_image = cv2.GaussianBlur(line_image, (35, 41), 0)

        _, binarizedImage = cv2.threshold(line_image, 252, 255, cv2.THRESH_BINARY)

        horizontal_projection = np.sum(binarizedImage, axis=0)
        horizontal_projection[horizontal_projection < np.max(horizontal_projection)] = 0
        horizontal_projection[horizontal_projection > 0] = 255

        words = []
        start_index = None
        for i, val in enumerate(horizontal_projection):
            if val == 0 and start_index is None:
                start_index = i
            elif val == 255 and start_index is not None:
                words.append(Word(start_index, i - 1, line.y1, line.y2))
                start_index = None

        if start_index is not None:
            words.append(Word(start_index, len(horizontal_projection) - 1, line.y1, line.y2))

        line.words = words

    return lines


def get_letters(image, lines):
    for line in lines:
        for word in line.words:
            word_image = image[word.y1:word.y2, word.x1:word.x2]
            word_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)

            _, binarizedImage = cv2.threshold(word_image, 240, 255, cv2.THRESH_BINARY)

            horizontal_projection = np.sum(binarizedImage, axis=0)
            horizontal_projection[horizontal_projection < np.max(horizontal_projection)] = 0
            horizontal_projection[horizontal_projection > 0] = 255

            letters = []
            start_index = None
            for i, val in enumerate(horizontal_projection):
                if val == 0 and start_index is None:
                    start_index = i
                elif val == 255 and start_index is not None:
                    letters.append(Letter(word.x1 + start_index, word.x1 + i - 1, word.y1, word.y2))
                    start_index = None

            if start_index is not None:
                letters.append(Letter(word.x1 + start_index, word.x1 + len(horizontal_projection) - 1, word.y1, word.y2))

            word.letters = letters

    return lines


if __name__ == '__main__':
    img = cv2.imread('text1_v3.png')
    lines = get_line_indices(img)
    lines = get_words(img, lines)
    lines = get_letters(img, lines)
    for line in lines:
        for word in line.words:
            for letter in word.letters:
                letter_img = img[letter.y1:letter.y2, letter.x1:letter.x2]
                letter_img = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
                letter.contour = get_contour(letter_img.astype(np.uint8))
    print('Done')
