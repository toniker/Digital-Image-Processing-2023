import numpy as np
import cv2


def getcountour(letter):
    cell_array = np.array((5, 2))

    kernel = np.ones((3, 3), np.uint8)

    letter = cv2.bitwise_not(letter)
    diff = cv2.morphologyEx(letter, cv2.MORPH_GRADIENT, kernel)

    ret, threshold = cv2.threshold(diff, 127, 255, 0)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del kernel, ret
    sequences = []

    for contour in contours:
        contour = np.squeeze(contour)
        sequence = []
        for i, point in enumerate(contour):
            sequence.append(point[0] + 1j * point[1])

        sequence = np.array(sequence)
        sequence_dft = np.fft.fft(sequence)
        sequence_dft_shift = np.fft.fftshift(sequence_dft)
        descriptor = np.abs(sequence_dft_shift[1::])
        sequences.append(descriptor)
        del sequence, sequence_dft, sequence_dft_shift, descriptor, contour, point

    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
    # contoured_image = cv2.drawContours(threshold, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("contoured.png", threshold)
    # sequences = np.array(sequences)

    return cell_array


if __name__ == "__main__":
    img = cv2.imread("a.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    c = getcountour(img)
    print("Done")
