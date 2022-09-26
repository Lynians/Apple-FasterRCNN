#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:12:16 2022

@author: yjla
"""
import os
import sys
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import transforms as tr
from torchvision import datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import imgaug as ia
from imgaug import augmenters as iaa
import time

import numpy as np
import pandas as pd

import cv2
from PIL import Image
import matplotlib.pyplot as plt


transforms_resize = tr.Compose([
             tr.ToPILImage(),
             tr.Resize((1152, 768)),  #tr.Resize((160, 90)),  # (h, w)   # PIL image in, PIL image out.
             #tr.ToTensor(),   # PIL image in, PyTorch tensor out.
             ])


def convert_from_cv2_to_pil(opencv_image: np.ndarray) -> Image:
    color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    pil_image = Image.fromarray(color_converted)  # convert from openCV2 to PIL

    return pil_image


def convert_from_pil_to_cv2(pil_image: Image) -> np.ndarray:
    array_converted = np.array(pil_image)
    cv_image = cv2.cvtColor(array_converted, cv2.COLOR_RGB2BGR)

    return cv_image


def displayPlt_image_mask_label(img_cv, msk_cv): 
    img_plt = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    msk_plt = cv2.cvtColor(msk_cv, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

    plt.figure(figsize=(24, 6))
    plt.subplot(141)
    plt.imshow(img_plt)
    plt.subplot(142)
    plt.imshow(msk_plt)
    
    plt.show()


def create_label_image(img_path, dataLabelFile):  
    pd_label = pd.read_csv(dataLabelFile[0]) 
    re_path = '/Apple211015/'

    bboxCoord = []
    centerCoord = []
    resBboxCoord = []
    resCenterCoord = []
    whs = []
    resWhs = []
    object_name = []
    # print(f'generate box mask : {idx}')
    # print(img_path)

    img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR : default value, (B, G, R)
    label_filename = os.path.basename(img_path)
    image_size = (img_cv.shape[0], img_cv.shape[1])
    pd_labels = pd_label[pd_label['filename'] == label_filename]
        
    region_count = pd_labels.iloc[0]['region_count']
    bbox = pd_labels['region_shape_attributes']

        
    for i in range(len(pd_labels)):
        row_dict = eval(bbox.iloc[i])  # eval() : transform str to dict
        xmin = row_dict['x']
        ymin = row_dict['y']
        width = row_dict['width']
        height = row_dict['height']
        xmax = int(xmin + width)
        ymax = int(ymin + height)
        
        xmin = (int(float(xmin)))
        ymin = (int(float(ymin)))
        xmax = (int(float(xmax)))
        ymax = (int(float(ymax)))
        bboxCoord.append([xmin, ymin, xmax, ymax])
        
        #return apple target
        object_name.append(1)
        
        
        center_x = int(np.around((xmin + xmax) / 2))
        center_y = int(np.around((ymin + ymax) / 2))
        centerCoord.append([center_x, center_y])
        whs.append([width, height])
           
    return img_path, object_name, bboxCoord

# image_size_cam1 = (5472, 3648)  # (H, W) (1.5,  1)
# image_size_cam2 = (3264, 2448)  # (H, W) (1.33, 1)
# image_size_resize = (1152, 768)  # 3:2 (768, 512)  (1024, 1024)
# in_channels = 3

dataImageList = glob(os.path.join('./AppleDataset/Train', '*.JPG'))
dataLabelFile = glob(os.path.join('./AppleDataset/Annotations', '*.csv'))
dataImageList.sort()

valImageList = glob(os.path.join('./AppleDataset/Validation', '*.JPG'))

# Check dataset
# for i in dataImageList:
#     path, objName, boxA = create_label_image(i, dataLabelFile)
#     print("path = ", path, "  obj = ", objName, "  box = ", boxA)


# path, objName, boxA = create_label_image(dataImageList[0], dataLabelFile)
# print(boxA[0])
    
    
# print(len(dataImageList), dataImageList)
# print(type(dataImageList))
    
label_dic = {}
label_dic['apple'] = 1
# create_label_image(dataImageList, dataLabelFile) 

########################
# #GPU연결
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)

########################


class AppleDataset(object):
    def __init__(self, transforms, dataList):
        self.transforms = transforms
        self.labelList = dataLabelFile
        self.imgs = dataList
        self.resize = iaa.Resize({"shorter-side" : 600, "longer-side":"keep-aspect-ratio"})
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_name, object_name, bbox = create_label_image(file_image, self.labelList)
        
        img = Image.open(self.imgs[idx]).convert("RGB")
        img = np.array(img)
        bbox = np.array(bbox)
        
        img, bbox = self.resize(image = img, bounding_boxes = bbox)
        bbox = bbox.squeeze(0).tolist()
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        targets = []
        d = {}
        d['boxes'] = torch.as_tensor(bbox)
        d['labels'] = torch.as_tensor(object_name, dtype = torch.int64)
        targets.append(d)
            
        return img, targets
        
data_transform = tr.Compose([
        tr.ToTensor()
    ])


def collate_fn(batch):
    return tuple(zip(*batch))

dataset = AppleDataset(data_transform, dataImageList)
val_dataset = AppleDataset(data_transform, valImageList)        

data_loader = DataLoader(dataset, batch_size = 1)
val_data_loader = DataLoader(val_dataset, batch_size=1)

print(next(iter(data_loader)))
## Model###

##Another Backbone##
# backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
# backbone.out_channels = 512

# anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),),
#                                     aspect_ratios=((0.5,1.0,2.0),))

# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], 
#                                                 output_size=7, 
#                                                 sampling_ratio=2)
# num_classes = 2 # apple + background

# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model = FasterRCNN(backbone,
#                     rpn_anchor_generator=anchor_generator,
#                     box_roi_pool=roi_pooler)




# print(model)
# model.to(device)
###################
# backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
# backbone_out = 512
# backbone.out_channels = backbone_out

# anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))

# resolution = 7
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

# box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= backbone_out*(resolution**2),representation_size=4096) 
# box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096,2) #21개 class

# model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
#                     min_size = 600, max_size = 1000,
#                     rpn_anchor_generator=anchor_generator,
#                     rpn_pre_nms_top_n_train = 6000, rpn_pre_nms_top_n_test = 6000,
#                     rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
#                     rpn_nms_thresh=0.7,rpn_fg_iou_thresh=0.7,  rpn_bg_iou_thresh=0.3,
#                     rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
#                     box_roi_pool=roi_pooler, box_head = box_head, box_predictor = box_predictor,
#                     box_score_thresh=0.05, box_nms_thresh=0.7,box_detections_per_img=300,
#                     box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
#                     box_batch_size_per_image=128, box_positive_fraction=0.25
#                   )

# # #roi head 있으면 num_class = None으로 함

# for param in model.rpn.parameters():
#   torch.nn.init.normal_(param,mean = 0.0, std=0.01)

# for name, param in model.roi_heads.named_parameters():
#   if "bbox_pred" in name:
#     torch.nn.init.normal_(param,mean = 0.0, std=0.001)
#   elif "weight" in name:
#     torch.nn.init.normal_(param,mean = 0.0, std=0.01)
#   if "bias" in name:
#     torch.nn.init.zeros_(param)

# model.to(device)
#######################
#Loss#
def Total_Loss(loss):
  loss_objectness = loss['loss_objectness']
  loss_rpn_box_reg = loss['loss_rpn_box_reg']
  loss_classifier = loss['loss_classifier']
  loss_box_reg = loss['loss_box_reg']

  rpn_total = loss_objectness + 10*loss_rpn_box_reg
  fast_rcnn_total = loss_classifier + 1*loss_box_reg

  total_loss = rpn_total + fast_rcnn_total

  return total_loss
########################
#Tensorboard code
writer = SummaryWriter("/home/yjla/Desktop/GPUBackup/Faster_RCNN2/runs")
########################
num_epochs = 400
len_data = 160
term = 160

loss_sum = 0
#Optimizer##
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum = 0.9, weight_decay = 0.0005)

#Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,eta_min=0.00001)

try:
  check_point = torch.load("/home/yjla/Desktop/GPUBackup/Faster_RCNN2/Check_point.pth") 
  start_epoch = check_point['epoch']
  start_idx = check_point['iter']
  model.load_state_dict(check_point['state_dict'])
  optimizer.load_state_dict(check_point['optimizer'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,eta_min=0.00001,last_epoch = start_epoch)
  scheduler.load_state_dict(check_point['scheduler'])

  if start_idx == len_data: 
    start_idx = 0
    start_epoch = start_epoch + 1

except:
  print("check point load error!")
  start_epoch = 0
  start_idx = 0

print("start_epoch = {} , start_idx = {}".format(start_epoch,start_idx))
print("Training Start")
model.train()
start = time.time()

# for epoch in range(start_epoch,num_epochs):
#     writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    
#     for i, (img,targets)in enumerate(data_loader, start_idx):
        
#         optimizer.zero_grad()
        
#         targets[0]['boxes'].squeeze_(0)
#         targets[0]['labels'].squeeze_(0)

#         loss = model(img, targets)
#         total_loss = Total_Loss(loss)
#         loss_sum += total_loss
        
#         if(i+1) % term == 0:
#             end = time.time()
#             print("Epoch {} | Iter {} | Loss: {} | Duration: {} min".format(epoch,(i+1),(loss_sum/term).item(),int((end-start)/60)))
#             writer.add_scalar('Training Loss',loss_sum / term, epoch * len_data + i)
      
#             state = {
#                 'epoch': epoch,
#                 'iter' : i+1,
#                 'state_dict': model.state_dict(),
#                 'optimizer' : optimizer.state_dict(),
#                 'scheduler': scheduler.state_dict()
#               }
#             torch.save(state,"/home/yjla/Desktop/GPUBackup/Faster_RCNN2/Check_point.pth")
     
#             loss_sum = 0
#             start = time.time()
    
#         total_loss.backward()
#         optimizer.step()

#     start_idx = 0
#     scheduler.step() 

#     state = {
#         'epoch': epoch,
#         'iter' : i+1,
#         'state_dict': model.state_dict(),
#         'optimizer' : optimizer.state_dict(),
#         'scheduler': scheduler.state_dict()
#         }
#     torch.save(state,"/home/yjla/Desktop/GPUBackup/Faster_RCNN2/Check_point.pth")

#     if (epoch+1) % 200 == 0: 
#         torch.save(model.state_dict(),"/home/yjla/Desktop/GPUBackup/Faster_RCNN2/Epoch{}.pth".format(epoch))