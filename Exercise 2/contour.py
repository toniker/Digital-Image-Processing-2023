import numpy as np
import cv2


def getcountour(letter):
    # Apply threshold to the image
    ret, binary_image = cv2.threshold(letter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary_image)
    del ret, binary_image

    # dilate the image
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inverted, kernel)

    # Remove the dilated image from the original image
    subtracted = cv2.subtract(letter, dilated)

    # Perform thinning of the result image
    inverted = cv2.bitwise_not(subtracted)
    eroded = cv2.erode(inverted, kernel)
    del inverted, dilated, subtracted, kernel

    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy.squeeze()

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
    del sequence, sequence_dft, sequence_dft_shift, descriptor, contour, point, i, hierarchy

    # Draw the contours
    # eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(eroded, contours, -1, (0, 255, 0), 1)
    # cv2.imwrite("contoured.png", eroded)

    return sequences


if __name__ == "__main__":
    img = cv2.imread("a.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    c = getcountour(img)
    print("Done")
