#Import image processing library
import cv2
import numpy as np
#Import libarary for file and directory handling
from pathlib import Path
#Import library for machine learning
import torch
from tqdm import tqdm
#Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print device
print("Device: ", device)

#Load the model from torch hub
#Repo:
repo = 'ultralytics/yolov5'
#Custom model:
model = 'custom'
#Model path:
model_path = 'best.pt'
model = torch.hub.load(repo, model, model_path)

#Input directory
input_dir = '/media/buntuml/DATASET/TEST_CASE/dam/wall/1'
# input_dir = '/media/buntuml/DATASET/DAMAGEAI/REPORT/spalling/DATASET_JSON/images'
#Create a list of all images in the directory
images = [x for x in Path(input_dir).glob('*.JPG')]
print(images[:10])
#loop through all images
for image in tqdm(images):
    #Get prediction from model
    prediction = model(image)
    #Get bounding boxes
    if len(prediction.xywh[0]) > 0:
        #Save prediction to file
        prediction.save('output')