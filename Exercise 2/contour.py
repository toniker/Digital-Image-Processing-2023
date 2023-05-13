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

    # Draw the contours
    # eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(eroded, contours, -1, (0, 255, 0), 1)
    # cv2.imwrite("contoured.png", eroded)

    return contours


def compare_contours(c1, c2):
    similarities = []
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

            if len(c1_sequence) > len(c2_sequence):
                c2_sequence = np.pad(c2_sequence, (0, len(c1_sequence) - len(c2_sequence)), 'constant', constant_values=(0, 0))
            elif len(c2_sequence) > len(c1_sequence):
                c1_sequence = np.pad(c1_sequence, (0, len(c2_sequence) - len(c1_sequence)), 'constant', constant_values=(0, 0))

            # Compute the difference between the two contours
            c1_sequence = np.abs(c1_sequence)
            c2_sequence = np.abs(c2_sequence)
            difference = c1_sequence - c2_sequence

            # Compute the similarity between the two contours
            similarity = np.sum(difference ** 2)
            similarities.append(similarity)

    return int(min(similarities))


if __name__ == "__main__":
    letters = ["a.png", "e.png", "f.png", "l.png"]
    letter_contours = {}
    for letter in letters:
        img = cv2.imread(letter)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sequence = getcountour(img)
        letter_contours[letter] = sequence

    # for letter, sequence in letter_contours.items():
    #     print(letter)
    #     for letter2, sequence2 in letter_contours.items():
    #         if letter != letter2:
    #             print(f"{letter} and {letter2} are {compare_contours(sequence, sequence2)} points similar")

    input_image = cv2.imread("input_a_text1.png")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_contour = getcountour(input_image)
    results = {}
    for letter, contour in letter_contours.items():
        result = compare_contours(input_contour, contour)
        print(f"{letter} and input are {result} points similar")
        results[letter] = result

    print(f"Input is most similar to {min(results, key=results.get)}")
