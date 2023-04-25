import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class VesselDataset(Dataset):
    def __init__(self, image_dir, mask_dir,transform=None,names=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.names = names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mask_dir != None:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index])
            # 若不加convert的话是RGBA四个通道，多了个透明通道
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)


            # 把 mask里255.0 的全部变成1.0
            mask[mask == 255 ] = 1.0
            #print('无tran：',image.shape)

            # 数据增强
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
                #print('tran后：',image.shape)
            else:
                # 涉及到一个归一化的问题，将np类型转化为tensor，并且修改一下rgb的顺序。
                image = torch.from_numpy(image.transpose((2, 0, 1)))
                mask = torch.from_numpy(mask)
                #print('改变后的image：',image.shape)

            if self.names:
                return image, mask,self.images[index]   

            return image, mask
        
        else:
            img_path = os.path.join(self.image_dir, self.images[index])
            # 若不加convert的话是RGBA四个通道，多了个透明通道
            image = np.array(Image.open(img_path).convert("RGB"))

            # 数据增强
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]
            else:
                # 涉及到一个归一化的问题，将np类型转化为tensor，并且修改一下rgb的顺序。
                image = torch.from_numpy(image.transpose((2, 0, 1)))

            if self.names:
                return image, self.images[index]   

            return image
