import os
import cv2
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split


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
    

if __name__ == "__main__":
    main()
