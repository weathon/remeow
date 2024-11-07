import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
import torch
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, train_path, val_path, fold, mode='train'):
        self.data_path = train_path if mode == 'train' else val_path
        self.fold = fold
        self.mode = mode

        txt_file = os.path.join(val_path, f'{mode}_{fold}.txt')
        with open(txt_file, 'r') as f:
            image_names = f.read().split("\n")

        if mode == 'train':
            self.image_names = []
            for name in image_names:
                for i in range(1, 6):
                    self.image_names.append(f'{i}_{name}')
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8),
                # transforms.RandomResizedCrop((512, 512)),
            ])
        else:
            self.image_names = random.sample(image_names, 512)
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8),
                transforms.Resize((512, 512)),
            ])

    def __len__(self):
        return len(self.image_names)

    def close(self, a, b):
        diff = np.abs(a - b)
        return diff < 0.04 * b
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        in_path = os.path.join(self.data_path, 'in', image_name)
        long_path = os.path.join(self.data_path, 'long', image_name)
        short_path = os.path.join(self.data_path, 'short', image_name)
        gt_path = os.path.join(self.data_path, 'gt', image_name) 
        in_path = os.path.join(self.data_path, 'in', image_name)

        in_image = np.array(Image.open(in_path).resize((512, 512), Image.NEAREST))
        long_image = np.array(Image.open(long_path).resize((512, 512), Image.NEAREST))
        short_image = np.array(Image.open(short_path).resize((512, 512), Image.NEAREST))
        gt_image = np.array(Image.open(gt_path).resize((512, 512), Image.NEAREST))
        if self.mode == 'train':
            roi_path = os.path.join(self.data_path, 'ROI', image_name)
            roi_image = np.array(Image.open(roi_path).resize((512, 512), Image.NEAREST))
        else:
            roi_image = ~self.close(gt_image.mean(-1), 85) * 255
        in_image = np.array(Image.open(in_path).resize((512, 512), Image.NEAREST))

        long_image, short_image, gt_image, in_image, roi_image = self.transform(long_image, short_image, gt_image, in_image, roi_image)

        X = torch.cat([in_image, long_image, short_image], dim=0)
        ROI = transforms.functional.resize(roi_image, (512, 512))
        Y = gt_image
        Y = transforms.functional.resize(Y, (512, 512))
        X, Y, ROI = X/255, Y/255, ROI/255
        Y = (Y > 0.95).float()
        return X.to(torch.float32), Y.to(torch.float32).mean(0), ROI.to(torch.float32).mean(0)
    
        # return {
        #     'in': in_image,
        #     'long': long_image,
        #     'short': short_image,
        #     'gt': gt_image,
        #     'roi': roi_image
        # }
    

# test

# if __name__ == '__main__':
#     import pylab
#     import numpy as np
#     import cv2
#     train_path = '/mnt/fastdata/preaug_cdnet/'
#     val_path = '/mnt/fastdata/CDNet/'
#     fold = 1
#     dataset = CustomDataset(train_path, val_path, fold, mode='val')
#     toshow = np.zeros((512, 512*5, 3), dtype=np.uint8)
#     for i, key in enumerate(dataset[1].keys()):
#         toshow[:, i*512:(i+1)*512] = np.array(dataset[5000][key].permute(1, 2, 0))
#     cv2.imwrite('test.png', toshow)
    
