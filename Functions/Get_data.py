import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
DATA_PATH = Path("D:/PROJECT")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it
if IMAGE_PATH.is_dir():
    print(f"{IMAGE_PATH} already exists")
else:
    print(f"Did not find {IMAGE_PATH} directory, creating one...")
    IMAGE_PATH.mkdir(parents=True, exist_ok=True)

# Download the data from the github account of Mr D. bourke
with open(DATA_PATH / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading data...")
    f.write(request.content)

# Unzip the data 
with zipfile.ZipFile(DATA_PATH / "pizza_steak_sushi.zip", "r") as zip_ref:
    print(f"Unzipping the data...")
    zip_ref.extractall(IMAGE_PATH)
    print("Done")
    
# Remove zip file 
#os.remove(DATA_PATH / "pizza_steak_sushi")     # If the access is denied then we can comment this code line 