import cv2
import numpy as np
from scipy.interpolate import interp1d


def get_contour(letter):
    # Apply threshold to the image
    ret, binary_image = cv2.threshold(letter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary_image)
    del ret, binary_image

    # Dilate the image
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inverted, kernel)
    # Remove the dilated image from the original image
    subtracted = cv2.subtract(letter, dilated)

    # Perform thinning of the result image
    inverted = cv2.bitwise_not(subtracted)
    eroded = cv2.erode(inverted, kernel)
    del inverted, dilated, subtracted, kernel

    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def compare_contours(c1, c2):
    """
    Compare two letters using their Fourier descriptors. The letters may have a different number of contours, and each
    pair will be compared. The result is the lowest sum of the differences of each pair of contours.
    :param c1: The contour of the first letter
    :param c2: The contour of the second letter
    :return: The lowest sum of the differences of each pair of contours
    """
    similarities = []
    if len(c1) != len(c2):
        return np.inf

    # Compare the contours
    for c11 in c1:
        contour = np.squeeze(c11)

        seq = []
        for i, point in enumerate(contour):
            seq.append(point[0] + 1j * point[1])

        seq = np.array(seq)
        sequence_dft = np.fft.fft(seq)
        sequence_dft_shift = np.fft.fftshift(sequence_dft)
        descriptor = np.abs(sequence_dft_shift[1::])
        c1_sequence = descriptor
        del seq, sequence_dft, sequence_dft_shift, descriptor

        for c22 in c2:
            contour = np.squeeze(c22)

            seq = []
            for i, point in enumerate(contour):
                seq.append(point[0] + 1j * point[1])

            seq = np.array(seq)
            sequence_dft = np.fft.fft(seq)
            sequence_dft_shift = np.fft.fftshift(sequence_dft)
            descriptor = np.abs(sequence_dft_shift[1::])
            c2_sequence = descriptor
            del seq, sequence_dft, sequence_dft_shift, descriptor

            # If we need to interpolate the sequences, they must be at least 4 points long
            if len(c1_sequence) != len(c2_sequence) and len(c1_sequence) < 4 or len(c2_sequence) < 4:
                similarities.append(9999999999999)
                continue

            # Interpolate the sequences to be the same length
            if len(c1_sequence) > len(c2_sequence):
                x = np.linspace(0, 1, len(c2_sequence))
                y = np.random.rand(len(c2_sequence))

                f = interp1d(x, y, kind="cubic")
                xnew = np.linspace(0, 1, len(c1_sequence))
                c2_sequence = f(xnew)
                del x, y, f, xnew
            elif len(c2_sequence) > len(c1_sequence):
                x = np.linspace(0, 1, len(c1_sequence))
                y = np.random.rand(len(c1_sequence))

                f = interp1d(x, y, kind="cubic")
                xnew = np.linspace(0, 1, len(c2_sequence))
                c1_sequence = f(xnew)
                del x, y, f, xnew

            # Compute the difference between the two contours
            c1_sequence = np.abs(c1_sequence)
            c2_sequence = np.abs(c2_sequence)
            difference = c1_sequence - c2_sequence

            # Compute the similarity between the two contours
            similarity = np.sum(difference ** 2)
            similarities.append(similarity)

    # Return the "score" of the most similar contour. Lower score means more similar
    return int(min(similarities))


if __name__ == "__main__":
    letters = ["a.png", "e.png", "f.png", "l.png"]
    letter_contours = {}
    for letter in letters:
        img = cv2.imread(letter)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sequence = get_contour(img)
        letter_contours[letter] = sequence

    # for letter, sequence in letter_contours.items():
    #     print(letter)
    #     for letter2, sequence2 in letter_contours.items():
    #         if letter != letter2:
    #             print(f"{letter} and {letter2} are {compare_contours(sequence, sequence2)} points similar")

    input_image = cv2.imread("f.png")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_contour = get_contour(input_image)
    results = {}
    for letter, contour in letter_contours.items():
        result = compare_contours(input_contour, contour)
        print(f"{letter} and input are {result} points similar")
        results[letter] = result

    print(f"Input is most similar to {min(results, key=results.get)}")
