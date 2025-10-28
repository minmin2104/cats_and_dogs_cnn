import os
import cv2
import numpy as np
import zipfile


EXTRACT_PATH = "./content/cats_dogs"


def unzip_content():
    with zipfile.ZipFile("./cats_and_dogs_mini_datasets.zip", "r") as zip_obj:
        zip_obj.extractall(EXTRACT_PATH)
        
    print("datasets unzip successfully")


def load_data_and_labels():
    categories = ["cats_set", "dogs_set"]
    data = []
    labels = []
    for ct in categories:
        path = os.path.join(EXTRACT_PATH, ct)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (32, 32))
            data.append(img)
            labels.append(0 if ct == "cats_set" else 1)
    data = np.array(data)
    labels = np.array(labels)

    data = data / 255.0
    print("Data loaded successfully!")
    print("Data shape:", data.shape)
    print("Label shape:", labels.shape)
    return data, labels

def main():
    data, label = load_data_and_labels()
    

if __name__ == "__main__":
    main()
