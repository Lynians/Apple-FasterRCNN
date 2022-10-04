import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from tqdm import tqdm


def train_model(dataloader, learning_rate, num_epochs, device):
    
    def get_total_loss(losses):
        loss_objectness = losses['loss_objectness']
        loss_rpn_box_reg = losses['loss_rpn_box_reg']
        loss_classifier = losses['loss_classifier']
        loss_box_reg = losses['loss_box_reg']

        rpn_total = loss_objectness + 10*loss_rpn_box_reg
        fast_rcnn_total = loss_classifier + 1*loss_box_reg
        total_loss = rpn_total + fast_rcnn_total
        # total_loss = rpn_total + loss_box_reg

        return total_loss
    
    # 모델 설정
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
    backbone_out = 512
    backbone.out_channels = backbone_out

    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))

    resolution = 7
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= backbone_out*(resolution**2),representation_size=4096) 
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096,2) #21개 class

    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                        min_size = 600, max_size = 1000,
                        rpn_anchor_generator=anchor_generator,
                        rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
                        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                        rpn_nms_thresh=0.7,rpn_fg_iou_thresh=0.7,  rpn_bg_iou_thresh=0.3,
                        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                        box_roi_pool=roi_pooler, box_head = box_head, box_predictor = box_predictor,
                        box_score_thresh=0.05, box_nms_thresh=0.7,box_detections_per_img=300,
                        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                        box_batch_size_per_image=128, box_positive_fraction=0.25
                    ).to(device)
    model.train()
    
    # optimizer & scheduler difinition
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)
    
    # 훈련 진행
    for epoch in range(num_epochs):
        start_time = time.time()
        loss = 0.0
        for inputs, box_coord_and_labels in dataloader:
            inputs = inputs.to(device)
            box_coord_and_labels['boxes'] = box_coord_and_labels['boxes'].squeeze(0).to(device)
            box_coord_and_labels['labels'] = box_coord_and_labels['labels'].squeeze(0).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, [box_coord_and_labels])
            
            total_loss = get_total_loss(outputs)
            total_loss.backward()
            optimizer.step()
            
            loss += total_loss.item() * inputs.size(0)
        
        scheduler.step()
            
        loss /= len(dataloader.dataset)
        print("Epoch {0:2} | Loss: {1:.4f} | Duration: {2:.4f} min".format(
            epoch + 1, loss, (time.time() - start_time) / 60
        ))

    return model