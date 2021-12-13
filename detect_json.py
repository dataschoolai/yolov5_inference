
#Import image processing library
import cv2
import numpy as np
#Import libarary for file and directory handling
from pathlib import Path
#Import library for machine learning
import torch
from tqdm import tqdm
import json
import shutil
#Create a function to make directory for dataset
def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

#Define a function to draw bounding box
def draw_box(image, box:list, color=(0, 255, 0), thickness=2):
    # If bbox is a list of coordinates
    if type(box) is list:
        box = np.array(box)
    # Convert bbox xywh to xyxy
    box[2] += box[0]
    box[3] += box[1]
    # Draw bounding box
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    
    return image


#Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print device
print("Device: ", device)

#Load the model from torch hub

#Create weights veriable and assign path of weights model
path_weights = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_weights)

#Input directory
input_dir = '/media/buntuml/DATASET/TEST_CASE/dam/wall/1'
#Convert input directory to pathlib 
input_dir = Path(input_dir)

#Create image list
image_list = list(input_dir.glob('*.JPG'))

#Create a dataset directory
DATASET_DIR = Path('dataset_json')
make_dirs(DATASET_DIR)

#Loop over the image list
for path in tqdm(image_list[10:]):
    #Get label id 
    label_id = path.stem.split('.')[0]
    #Read image
    image = cv2.imread(str(path))
    #Get prediction from model
    prediction = model(image)
    #Get bounding boxes
    if prediction.pred[0].shape[0]:
        #Copy image to dataset directory
        input_img_path = path 
        output_img_path = Path(DATASET_DIR)/'images'/f'{path.stem}.jpg'
        shutil.copy(input_img_path,output_img_path)


        #Get image width and height
        height, width = image.shape[:2]

        #Create a annotations dictionary
        annotations = {
            "annotations": [],
        }
        #Add images key to annotations dictionary
        annotations["images"] = [{
            'id':1,
            'file_name': str(output_img_path.name),
            'height': height,
            'width': width

        }]
        #Loop over the prediction
        for pred in prediction.xywh[0]:
            #Get bounding box: [x_center, y_center, width, height]
            x_center, y_center, width, height,class_id,confidence = pred
            #Convert x_center, y_center, width, height to xywh
            x_min = int(x_center - width/2)
            y_min = int(y_center - height/2)
            width = int(width)
            height = int(height)
            #Create a bounding box list 
            bbox = [x_min, y_min, width, height]
            #Get class
            class_id = int(pred[4])
            #get confidence
            confidence = float(pred[5])
            
            #add bounding box to annotations dictionary
            annotations["annotations"].append({
                "id": 1,
                "image_id": 1,
                "category_id": class_id,
                "bbox": bbox,
                "score": confidence
                })
            img = draw_box(image, bbox)
            #Save image
            cv2.imwrite(str(output_img_path), img)
        #Create a label dictionary

        print(annotations)
        prediction.files = [path.name]
        #Save prediction to file
        prediction.save('output')
        #Save annotations to json file
        with open(f'{DATASET_DIR}/labels/{label_id}.json','w',encoding='utf8') as f:        
            json.dump(annotations, f, ensure_ascii=False)

