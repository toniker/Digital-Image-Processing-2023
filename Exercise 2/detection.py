import cv2
import numpy as np

from contour import get_contour, compare_contours


class KnownLetter:
    def __init__(self, name, contour):
        self.name = name
        self.contour = contour


class Letter:
    def __init__(self, x1, x2, y1, y2, contour=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.coordinates = ((x1, y1), (x2, y2))
        self.contour = contour
        self.looks_like = None


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
                letters.append(
                    Letter(word.x1 + start_index, word.x1 + len(horizontal_projection) - 1, word.y1, word.y2))

            word.letters = letters

    return lines


if __name__ == '__main__':
    img = cv2.imread('letters.png')
    lines = get_line_indices(img)
    lines = get_words(img, lines)
    lines = get_letters(img, lines)

    known_letter_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                          'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                          'w', 'x', 'y', 'z']
    known_letters = []
    for known_letter_name in known_letter_names:
        known_letter_img = cv2.imread('letters/' + known_letter_name + '.png')
        known_letter_img = cv2.cvtColor(known_letter_img, cv2.COLOR_BGR2GRAY)
        known_letter_contour = get_contour(known_letter_img.astype(np.uint8))
        known_letters.append(KnownLetter(known_letter_name, known_letter_contour))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del known_letter_names, known_letter_name, known_letter_img, known_letter_contour
    string = ''
    for line in lines:
        for word in line.words:
            for letter in word.letters:
                letter_img = img[letter.y1:letter.y2, letter.x1:letter.x2]
                letter.contour = get_contour(letter_img.astype(np.uint8))
                best_score = np.inf
                for known_letter in known_letters:
                    score = compare_contours(letter.contour, known_letter.contour)
                    if score < best_score:
                        best_score = score
                        letter.looks_like = known_letter
                if best_score == np.inf:
                    letter.looks_like = KnownLetter('?', None)

                string += letter.looks_like.name
            string += ' '
        string += '\n'

    print(string)
    print('Done')
