import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models

from PIL import Image


class AppleDataset(Dataset):
    def __init__(self, file_paths, box_coords_and_labels, init_transforms=None):
        if(init_transforms == None):
            self.transforms = transforms.ToTensor()
        else:
            self.transforms = init_transforms
        self.file_paths = file_paths
        self.box_coords_and_labels = box_coords_and_labels
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        image = self.transforms(Image.open(self.file_paths[idx]).convert("RGB"))
        box_coord_and_label = self.box_coords_and_labels[idx]
        
        return image, box_coord_and_label