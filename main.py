import os
import cv2
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


EXTRACT_PATH = "./content/cats_dogs"


def unzip_content():
    with zipfile.ZipFile("./cats_and_dogs_mini_datasets.zip", "r") as zip_obj:
        zip_obj.extractall(EXTRACT_PATH)
        
    print("datasets unzip successfully")


def load_data_and_labels():
    categories = ["cats_set", "dogs_set"]
    data = []
    label = []
    for ct in categories:
        path = os.path.join(EXTRACT_PATH, ct)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (32, 32))
            data.append(img)
            label.append(0 if ct == "cats_set" else 1)
    data = np.array(data)
    label = np.array(label)

    data = data / 255.0
    return data, label

def main():
    print("Loading and normalizing data...")
    data, label = load_data_and_labels()
    print("Data loaded successfully!")
    print("Data shape:", data.shape)
    print("Label shape:", label.shape)

    # Splitting training data and testing data
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    print("Successfully split data")

    print("Using TensorFlow version:", tf.__version__)

    x_train = x_train.reshape(-1, 32, 32, 1)
    x_test = x_test.reshape(-1, 32, 32, 1)

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    model = keras.Sequential([
        # First convolution + pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D(2, 2),
        # Second convolution + pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # Flatten layer converts 2D feature maps → 1D vector
        layers.Flatten(),
        # Fully connected layer
        layers.Dense(64, activation='relu'),
        # Output layer → 1 neuron (0 or 1)
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    # Evaluate the Model on Test Data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("\n Test Accuracy:", test_acc)


if __name__ == "__main__":
    print("Starting main...")
    main()
