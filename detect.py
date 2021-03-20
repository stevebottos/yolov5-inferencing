import argparse
import time
from copy import copy
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(model, weights, device, image, imgsz = 640):  

    # Redefining opts here
    im0 = image
    img = image
    project = "runs/detect"
    name = "exp"
    save_img = True
    conf_thresh = 0.25
    iou_thresh = 0.45
    classes = [0] # Filters our only people

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)
    # img = letterbox(image, imgsz, stride)[0]
    
    # images need to be in (BATCHSIZE, DEPTH, W, H)
    
    if img.shape[0] == img.shape[1] and img.shape[2] == 3:
        img = np.rollaxis(img, 0, 3)
        img = np.rollaxis(img, 0, 3)

    batch_of_one = np.array([img])
    img = torch.from_numpy(batch_of_one).to(device).float() / 255.0

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=True))  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=classes, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        
    return det.detach().cpu().numpy()