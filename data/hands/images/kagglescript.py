import kaggle as kg
import kagglehub
import pandas as pd
import os

os.environ['KAGGLE_USERNAME'] = 'jl32111'
os.environ['KAGGLE_KEY'] = '1f612b2d5949489d53b6358184ee5d80'

myApi = kg.api
myApi.authenticate()

# myApi.dataset_download_files('moltean/fruits', path='./images')

# doesnt work
# path = kagglehub.dataset_download("moltean/fruits", path="./images")

# print("Path to dataset files:", path)
