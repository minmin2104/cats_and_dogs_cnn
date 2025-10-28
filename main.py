import zipfile


EXTRACT_PATH = "./content/cats_dogs"


def unzip_content():
    with zipfile.ZipFile("./cats_and_dogs_mini_datasets.zip", "r") as zip_obj:
        zip_obj.extractall(EXTRACT_PATH)
        
    print("datasets unzip successfully")


def main():
    pass
    

if __name__ == "__main__":
    main()
