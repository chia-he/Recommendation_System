import argparse
import torch
import torchvision
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import numpy as np
import cv2 
import time
import warnings
warnings.filterwarnings("ignore")

from model.GazeFollow import ModelSpatial
from py_utils import imutils, evaluation, visualization
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='./model/model_demo.pt')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target decision threshold', default=100)
args = parser.parse_args()

def frcnn_detect(model, img, threshold):

    # Transforms
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = img.cuda() # With Gpu

    # Predictions
    pred = model([img]) 
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]

    # masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy() # For MaskRCNN
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    
    # masks = masks[:pred_t+1] # For MaskRCNN
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    # return masks, pred_boxes, pred_class # For MaskRCNN
    return pred_boxes, pred_class

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def fetchData(filename, outputName, record=None, stride=None, device='cpu', mode='Low'):

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.empty_cache() # Clear cache

    # Load MTCNN model
    mtcnn = MTCNN(device=device) # For face detection

    # Load FasterCNN model
    if mode == 'High':
        #frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        frcnn_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    else:
        frcnn_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True) # Object detection (Low resolution, Fast, Low accuracy), call fasterrcnn_mobilenet_v3_large_fpn(pretrained=True) for Full resolution, Slow, High accuracy

    # Load attention target model
    test_transforms = _get_transform() # set up data transformation
    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights) # With Gpu
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    if device == 'cuda':
        frcnn_model.cuda() # With Gpu
        model.cuda() # With Gpu
    
    frcnn_model.eval()
    model.train(False)
    
    count = 0
    prev_frame_time = 0 # used to record the time when we processed last frame
    new_frame_time = 0 # used to record the time at which we processed current frame
    
    cap = cv2.VideoCapture(f'./data/input/{filename}')

    ret, frame = cap.read()

    # Not resized
    #size = (int(frame.shape[1]), int(frame.shape[0]))
    size = (400, 240)
    
    frame = cv2.resize(frame, size) # New

    vid = cv2.VideoWriter(f'./data/output/{outputName}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1], frame.shape[0]))
    
    with torch.no_grad():

        while True:
            
            new_frame_time = time.time()

            ret, frame = cap.read()
            
            # Read Failed : No Frame
            if not ret:
                break

            frame = cv2.resize(frame, size)

            count += 1
            
            if not (stride is None):
                if count % stride < (stride-1):
                    continue
            
            print("Frame:", count)
            try: 
                # Detect face
                boxes_face, probs, landmarks = mtcnn.detect(frame, landmarks=True)
                
                # enlarge the head box
                boxes_face[0][0], boxes_face[0][1], boxes_face[0][2], boxes_face[0][3] = boxes_face[0][0]*0.95, boxes_face[0][1]*0.95, boxes_face[0][2]*1.05, boxes_face[0][3]*1.05 # left, top, right, bottom

            except TypeError:
                #print("No faces detected!! or Exception!")
                continue
            
            # Convert OpenCV to PIL.Image
            frame_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_raw = Image.fromarray(frame_raw)
            
            width, height = frame_raw.size

            head_box = [boxes_face[0][0], boxes_face[0][1], boxes_face[0][2], boxes_face[0][3]]
            head_crop = frame_raw.crop((head_box)) 
            head_tf = test_transforms(head_crop)
            frame_tf = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], 
                                                        width, height,
                                                        resolution=input_resolution).unsqueeze(0)
            
            if device == 'cuda':
                head_uq = head_tf.unsqueeze(0).cuda() # With Gpu .cuda()
                frame_uq = frame_tf.unsqueeze(0).cuda() # With Gpu .cuda()
                head_channel_uq = head_channel.unsqueeze(0).cuda() # With Gpu .cuda()
            else:
                head_uq = head_tf.unsqueeze(0)
                frame_uq = frame_tf.unsqueeze(0)
                head_channel_uq = head_channel.unsqueeze(0)

            # Forward pass
            raw_hm, _, inout = model(frame_uq, head_channel_uq, head_uq)
            
            # Heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255

            target = (0, 0) # Default target position

            # 1. Arrow mode only shows in-frame gazes ( I removed the Heatmap option, check for demo.py in the github folder for heatmap implementation )
            # 2. output_resolution, input_resolution: variables from config.py

            if inout < args.out_threshold:
                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                target = (round(norm_p[0]*width), round(norm_p[1]*height))
            else:
                # make it looks-like always focus on target
                continue
                
            # Detect Object
            boxes_obj, pred_class = frcnn_detect(frcnn_model, frame_raw, 0.6)

            # Draw head and attention target in cv2
            try:
                if not (record is None):
                    record.drawTarget(image=frame, target=target, boxes=boxes_face, probs=probs, landmarks=landmarks) # New
            
            except cv2.error:
                #print("drawTarget Exception!!\n")
                pass

            # Draw Objects
            try :
                if not (record is None):
                    record.drawObjects(image=frame, target=target, boxes=boxes_obj, pred_objects=pred_class)

            except cv2.error:
                #print('No Instance to display!!')
                pass

            time_taken = new_frame_time-prev_frame_time
            fps = round(1/time_taken)
            prev_frame_time = new_frame_time
            
            w, h = Image.fromarray(frame).size
            coord = (0, int(h*0.085))

            cv2.putText(frame, 
                        "fps: {} time: {:.2f}s frame: {}".format(fps, time_taken, count), 
                        coord, cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 50, 0), 2, 
                        lineType=cv2.LINE_AA)
            cv2.imshow('Face Detection', frame)
            vid.write(frame)

            if not (record is None):
                record.rec_efficiency.append([count, fps, time_taken])
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        vid.release()
        cv2.destroyAllWindows()
        print('Finished!!')

    return record

if __name__ == "__main__":
    
    filename = "input2.avi"
    current = 'GPU_High_ResNet50'
    
    rec_objects = ['bottle', 'bowl', 'cup']
    record = visualization.Visual(rec_objects=rec_objects,
                                all_objects=COCO_INSTANCE_CATEGORY_NAMES)

    rd = fetchData(filename=filename, outputName=current, record=record, stride=None, mode='High')
    rd.outputData(sheet_name=current,columns=["Frame", "FPS", "Time Cost"])
