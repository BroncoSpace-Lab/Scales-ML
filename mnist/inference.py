from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(10, activation='softmax'),
    ])

    model.load_weights('cnn.weights.h5')

    predictions = model.predict(test_images)

    predicted_labels = np.argmax(predictions, axis=1)

    correct_predictions = np.sum(predicted_labels == test_labels)
    total_examples = test_images.shape[0]
    accuracy = correct_predictions / total_examples
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main()