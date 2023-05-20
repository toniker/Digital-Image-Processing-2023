import random

import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from detection import get_line_indices, get_words, get_letters


def get_chars(file):
    chars = []
    for line in file:
        for char in line:
            if char == ' ' or char == '\n':
                continue
            chars.append(char)

    return chars


if __name__ == "__main__":
    image = cv2.imread("text1_v3.png")
    text_file = open("text1_v3.txt", "r")
    chars_array = get_chars(text_file)
    image_data = get_line_indices(image)
    image_data = get_words(image, image_data)
    image_data = get_letters(image, image_data)

    y = chars_array
    x = []
    for line in image_data:
        for word in line.words:
            for letter in word.letters:
                letter_image = image[letter.y1:letter.y2, letter.x1:letter.x2]
                resized = cv2.resize(letter_image, (32, 32), interpolation=cv2.INTER_LINEAR_EXACT)
                _, binarizedImage = cv2.threshold(resized, 220, 255, cv2.THRESH_BINARY)
                binarizedImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                x.append(binarizedImage.flatten())

    for i in range(len(x)):
        cv2.imwrite(f'lines/{y[i]}.png', x[i].reshape(32, 32))
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random.randint(0, 100))

    # Create a KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=2)  # Specify the number of neighbors (K)

    # Train the classifier using the training data
    knn.fit(x_train, y_train)

    # Predict the labels for the test data
    y_pred = knn.predict(x_test)

    # Evaluate the accuracy of the classifier
    accuracy = knn.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")

    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    results = []
    for input_letter in letters:
        test_letter = cv2.imread(f'letters3/{input_letter}.png')
        test_letter = cv2.resize(test_letter, (32, 32), interpolation=cv2.INTER_CUBIC)
        _, test_letter = cv2.threshold(test_letter, 220, 255, cv2.THRESH_BINARY)
        test_letter = cv2.cvtColor(test_letter, cv2.COLOR_BGR2GRAY)
        prediction = knn.predict(test_letter.reshape(-1, test_letter.size))[0]
        results.append((input_letter, prediction))

    score = 0
    for result in results:
        if result[0] == result[1]:
            score += 1
        print(f"Input letter: {result[0]}, Prediction: {result[1]}")
    print(f"Accuracy on letters.png: {score / len(results)}")
