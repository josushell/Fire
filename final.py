##########################################################################

# Example : perform live fire detection in image/video/webcam using
# NasNet-A-OnFire, ShuffleNetV2-OnFire CNN models.

# Copyright (c) 2020/21 - William Thompson / Neelanjan Bhowmik / Toby
# Breckon, Durham University, UK

# License :
# https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn/blob/main/LICENSE

##########################################################################

import cv2
import os
import sys
import math
from PIL import Image
import argparse
import time
import numpy as np
import math

##########################################################################

import torch
import torchvision.transforms as transforms
from models import shufflenetv2
import torch.nn.utils.prune as prune

##########################################################################

# read/process image and apply tranformation


def read_img(frame, np_transforms,device):
    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    small_frame = Image.fromarray(small_frame)
    small_frame = np_transforms(small_frame).float()
    small_frame = small_frame.unsqueeze(0)
    small_frame = small_frame.to(device)

    return small_frame

##########################################################################

# model prediction on image


def run_model_img( frame, model):
    output = model(frame)
    pred = torch.round(torch.sigmoid(output))
    return pred

# def draw_pred(frame, prediction):
#     print(frame.shape)
#     height, width, _ = frame.shape
#     if prediction == 1:
#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)
#         cv2.putText(frame, 'No-Fire', (int(width / 16), int(height / 4)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     else:
#         cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
#         cv2.putText(frame, 'Fire', (int(width / 16), int(height / 4)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     return frame
def draw_pred(frame, prediction):

    fgbg = cv2.createBackgroundSubtractorMOG2()

    kernel = np.ones((5,5),np.uint8)
    height, width,_  = frame.shape

    if prediction == 1:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)
        cv2.putText(frame, 'No-Fire', (int(width / 16), int(height / 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame
    else:
        frame1 = cv2.pyrDown(frame) 
        fgmask = fgbg.apply(frame) 

        vid = cv2.medianBlur(fgmask,7)
        prdown = cv2.pyrDown(vid)

        vid1 = cv2.bitwise_and(frame1,frame1,mask=prdown)

        hsv = cv2.cvtColor(vid1,cv2.COLOR_BGR2HSV)

        lower = [5, 50, 50]
        upper = [100, 255, 255]

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)

        output = cv2.bitwise_and(vid1, hsv, mask=mask)
        no_red = cv2.countNonZero(mask)

        imgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray,127,255,0)
        result = cv2.dilate(thresh,kernel,iterations = 3) # 팽창

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(result)

        for index, centroid in enumerate(centroids):
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue

        x, y, width, height, area = stats[index]

        cv2.rectangle(frame1, (x, y), (x + width, y + height), (0, 255, 0),3)
        # cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
        # cv2.putText(frame, 'Fire', (int(width / 16), int(height / 4)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame1
##########################################################################
 
# uses cuda if available


def returnImage(im):
    
    device = torch.device('cpu')

    model = shufflenetv2.shufflenet_v2_x0_5(
        pretrained=False, layers=[
            4, 8, 4], output_channels=[
            24, 48, 96, 192, 64], num_classes=1)

    w_path = './weights/prunesff.pt'

    for name,module in model.named_modules():
        if 'branch' in name:
            if isinstance(module,torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.0)
    model.load_state_dict(torch.load(w_path, map_location=device))

    model.eval()
    model.to(device)

    frame = cv2.imread(im)

    np_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    im = read_img(frame, np_transforms,device)
    prediction = run_model_img( im, model)
    frame = draw_pred(frame, prediction)
    
    cv2.imwrite(f'./static/img/output/output.png',frame)

    return prediction

# model load

def result(root):
    print(returnImage(root))
    ## 0이면 불이고 1이면 아니다


