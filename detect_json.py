
#Import image processing library
import cv2
from PIL import Image
import numpy as np
#Import libarary for file and directory handling
from pathlib import Path
#Import library for machine learning
import torch
from tqdm import tqdm
import json
import shutil
#Import library to read command line arguments
import argparse

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

#Define a function predict the bounding box of the image
def predict(input_dir:Path,output_dir:Path,path_weights='best.pt'):

    #Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print device
    print("Device: ", device)

    #Load the model from torch hub

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_weights)

    #Convert input directory to pathlib 
    input_dir = Path(input_dir)
    #Create images extension list
    images_ext = ['.JPG','.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dng']
    #Get images list
    image_list = []
    for ext in images_ext:
        image_list.extend(list(input_dir.glob('*'+ext)))

   

    #Loop over the image list
    for path in tqdm(image_list[10:]):
        #Get label id 
        label_id = path.stem.split('.')[0]
        #Read image
        # image = cv2.imread(str(path))
        image = Image.open(path)
        #Get prediction from model
        prediction = model(image)
        #Get bounding boxes
        if prediction.pred[0].shape[0]:
            #Copy image to dataset directory
            input_img_path = path 
            output_img_path = Path(output_dir)/'images'/f'{path.stem}.jpg'
            shutil.copy(input_img_path,output_img_path)


            #Get image width and height
            width, height = image.size
            

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
                x_center, y_center, width, height,confidence,class_id = pred
                #Convert x_center, y_center, width, height to xywh
                x_min = int(x_center - width/2)
                y_min = int(y_center - height/2)
                width = int(width)
                height = int(height)
                #Create a bounding box list 
                bbox = [x_min, y_min, width, height]
                
                #add bounding box to annotations dictionary
                annotations["annotations"].append({
                    "id": 1,
                    "image_id": 1,
                    "category_id": str(class_id.item()),
                    "bbox": bbox,
                    "score": float(confidence)
                    })
            #Create a label dictionary

            prediction.files = [path.name]
            #Save prediction to file
            prediction.save('output')
            #Save annotations to json file
            with open(f'{output_dir}/labels/{label_id}.json','w',encoding='utf8') as f:        
                json.dump(annotations, f, ensure_ascii=False)
#Define the function to parse command line arguments
def parse_args():
    #Create a parser
    parser = argparse.ArgumentParser(description='Detect objects in images')
    #Add arguments
    parser.add_argument('--input', type=str,required=True, help='input directory')
    parser.add_argument('--output', type=str, default='output', help='output directory')
    parser.add_argument('--weights', type=str, default='best.pt', help='path of weights')
    #Return arguments
    return parser.parse_args()

#Run the main function
if __name__ == '__main__':
    #Parse command line arguments
    args = parse_args()
    #Get input and output directory
    input_dir = args.input
    output_dir = args.output
    #Get path of weights
    path_weight = args.weights

    #Create directory
    make_dirs(output_dir)
    #Predict bounding box
    predict(input_dir=input_dir,output_dir=output_dir,path_weights=path_weight)