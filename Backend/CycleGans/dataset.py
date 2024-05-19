from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HazeDehazeDataset(Dataset):       
    def __init__(self, root_dehaze, root_haze, transform=None,split=0.8):
        self.root_dehaze = root_dehaze
        self.root_haze = root_haze
        self.transform = transform
        self.split = split

        self.dehaze_images = os.listdir(root_dehaze)
        self.haze_images = os.listdir(root_haze)
        self.length_dataset = max(len(self.dehaze_images), len(self.haze_images)) # 1000, 1500
        self.dehaze_len = int(len(self.dehaze_images) * split)
        self.haze_len = int(len(self.haze_images) * split)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        dehaze_img = self.dehaze_images[index % self.dehaze_len]
        haze_img = self.haze_images[index % self.haze_len]

        dehaze_path = os.path.join(self.root_dehaze, dehaze_img)
        haze_path = os.path.join(self.root_haze, haze_img)

        dehaze_img = np.array(Image.open(dehaze_path).convert("RGB"))
        haze_img = np.array(Image.open(haze_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=dehaze_img, image0=haze_img)
            dehaze_img = augmentations["image"]
            haze_img = augmentations["image0"]

        return dehaze_img, haze_img