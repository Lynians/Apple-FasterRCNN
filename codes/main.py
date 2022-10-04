import os
import sys
from glob import glob

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms as tr
from torchvision import datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import time

import numpy as np
import pandas as pd

import cv2
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from csv_to_numpy import get_csv_data
from dataset import AppleDataset
from train import train_model


# image_size_cam1 = (5472, 3648)  # (H, W) (1.5,  1)
# image_size_cam2 = (3264, 2448)  # (H, W) (1.33, 1)


if __name__ == "__main__":
    train_folder = r'./AppleDataset/Train'
    val_folder = r'./AppleDataset/Validation'
    csv_file_path = r'./AppleDataset/Annotations/Apple21.csv'
    
    # device num (GPU number)
    device_num = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print("GPU_number : ", device_num, '\tGPU name:', torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # hyper parameters
    learning_rate = 1e-4
    batch_size = 1  # 1이 아닐 시 에러
    num_epochs = 100
    num_classes = 1
    # resize_shape = (600, 800)
    resize_shape = None
    
    # train val 파일 리스트, bounding box GT 가져오기
    file_paths, box_coords_and_labels = get_csv_data(csv_file_path, train_folder, val_folder, resize_shape)
    
    # dataset, dataloader 생성
    my_transforms = transforms.Compose([
        # transforms.Resize(resize_shape),
        transforms.ToTensor()
    ])
    
    apple_datasets = {}
    dataloaders = {}
    
    for phase in {'train', 'val'}:
        apple_datasets[phase] = AppleDataset(file_paths[phase], box_coords_and_labels[phase], my_transforms)
        dataloaders[phase] = DataLoader(apple_datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
    # 훈련
    model = train_model(dataloaders['train'], learning_rate, num_epochs, device)