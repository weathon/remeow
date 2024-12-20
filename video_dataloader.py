import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
import torch
import numpy as np
import random
import torchvision
from transformers import AutoImageProcessor
IMG_SIZE = 512
image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
class CustomDataset(Dataset):
    def __init__(self, train_path, val_path, args, mode='train'):
        self.data_path = train_path if mode == 'train' else val_path
        fold = args.fold
        self.args = args
        self.fold = fold
        self.mode = mode

        txt_file = os.path.join(val_path, f'{mode}_{fold}.txt')
        with open(txt_file, 'r') as f:
            image_names = sorted(f.read().split("\n"))
        self.mean = torch.tensor([0.485, 0.456, 0.406] * 3)
        self.std = torch.tensor([0.229, 0.224, 0.225] * 3)
        if mode == 'train':
            self.image_names = image_names 
            self.transform = transforms.Compose([
                transforms.ToImage(),
            ])
            self.bg_trans = transforms.Compose([
                 transforms.RandomApply([
                    transforms.GaussianBlur((3,13), 1), 
                    ], 0.5),
                    transforms.RandomApply([
                        transforms.RandomRotation((-10,10)),
                    ], 0.5),
                    transforms.RandomApply([ 
                        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.98, 1.02)),
                    ], 0.5),

            ])
        else:
            self.image_names = sorted(random.sample(image_names, min(1024, len(image_names))))
            # self.image_names = image_names
            self.transform = transforms.Compose([
                transforms.ToImage(),

            ])
        self.noise = torchvision.transforms.v2.GaussianNoise(0.1)


    def crop(self, in_image, long_image, short_image, gt_image, roi_image):
        top = random.randint(0, IMG_SIZE//2)
        left = random.randint(0, IMG_SIZE//2) 
        width = random.randint(IMG_SIZE//2, IMG_SIZE)
        aspect_ratio = random.uniform(0.5, 1.5)
        height = int(width * aspect_ratio)
        
        width = min(width, IMG_SIZE - left)
        height = min(height, IMG_SIZE - top)

        in_image = torchvision.transforms.functional.resized_crop(in_image, top, left, height, width, (IMG_SIZE, IMG_SIZE))
        long_image = torchvision.transforms.functional.resized_crop(long_image, top, left, height, width, (IMG_SIZE, IMG_SIZE))
        short_image = torchvision.transforms.functional.resized_crop(short_image, top, left, height, width, (IMG_SIZE, IMG_SIZE))
        gt_image = torchvision.transforms.functional.resized_crop(gt_image, top, left, height, width, (IMG_SIZE, IMG_SIZE))
        roi_image = torchvision.transforms.functional.resized_crop(roi_image, top, left, height, width, (IMG_SIZE, IMG_SIZE))
        return in_image, long_image, short_image, gt_image, roi_image
    
    
    
    def __len__(self):
        return len(self.image_names)

    def strong_pan(self, img): 
        shifteds = [img] 
        img2 = img
        for i in range(random.randint(5, 60)):
            shiftx = random.randint(0, 20)
            img = torchvision.transforms.functional.affine(img, angle=0, translate=(shiftx, 0), scale=1, shear=0)
            img2 = torchvision.transforms.functional.affine(img2, angle=0, translate=(-shiftx, 0), scale=1, shear=0)
            shifteds.append(img) 
            shifteds.append(img2)

        return torch.stack(shifteds).to(torch.float).mean(0)
            
    def weak_pan(self, img):
        shifteds = [img] 
        img2 = img
        for i in range(random.randint(5, 20)):
            shiftx = random.randint(0, 20)
            img = torchvision.transforms.functional.affine(img, angle=0, translate=(shiftx, 0), scale=1, shear=0)
            img2 = torchvision.transforms.functional.affine(img2, angle=0, translate=(-shiftx, 0), scale=1, shear=0)
            shifteds.append(img) 
            shifteds.append(img2)

        return torch.stack(shifteds).to(torch.float).mean(0)
    
    def close(self, a, b):
        diff = np.abs(a - b) 
        return diff < 0.05 * b
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        long_path = os.path.join(self.data_path, 'long' if self.args.background_type == "mog2" else "sub_long", image_name)
        short_path = os.path.join(self.data_path, 'short' if self.args.background_type == "mog2" else "sub_short", image_name)
        gt_path = os.path.join(self.data_path, 'gt', image_name) 

        long_image = np.array(Image.open(long_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        short_image = np.array(Image.open(short_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        gt_image = np.array(Image.open(gt_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        roi_image = ~self.close(gt_image.mean(-1), 85) * 255

        long_image, short_image, gt_image, roi_image = self.transform(long_image, short_image, gt_image, roi_image)
        if self.mode == 'train':
            if random.random() > 0.9:
                long_image = self.strong_pan(long_image)
                short_image = self.weak_pan(short_image) 
            if random.random() > 0.9:
                long_image = long_image + long_image * (torch.rand(1) * 0.5 - 0.25)
            if random.random() > 0.9:
                short_image = short_image + short_image * (torch.rand(1) * 0.5 - 0.25)
            if random.random() > 0.7:
                short_image = self.bg_trans(short_image)
            if random.random() > 0.7:
                long_image = self.bg_trans(long_image)
        in_images = []
        for i in range(0,80,10):
            image_id = int(image_name.split("_")[-1].split(".")[0].replace("in","")) - i
            image_id = str(image_id).zfill(6)
            in_image_path = os.path.join(self.data_path, 'in', "_".join(image_name.split("_")[:-1] + [f"in{image_id}.jpg"]))
            in_image = np.array(Image.open(in_image_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
            in_image = self.transform(in_image)
            in_image = image_processor(images=in_image/max(255, in_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
            in_images.append(in_image)
            
        in_images = torch.cat(in_images, dim=0)
        long_image = image_processor(images=long_image/max(255, long_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
        short_image = image_processor(images=short_image/max(255, short_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
        X = torch.cat([in_images, long_image, short_image], dim=0)
        if self.mode == "train":
            if random.random() > 0.7:
                X = X + X * (torch.rand(X.shape[0])[:,None,None] * 0.5 - 0.25)
                
        ROI = transforms.functional.resize(roi_image, (IMG_SIZE, IMG_SIZE))
        Y = gt_image
        Y = transforms.functional.resize(Y, (IMG_SIZE, IMG_SIZE))
        Y = Y/255
        ROI = ROI/255
        ROI[0,0]=1
        ROI[0,1]=0
        if self.mode == 'train':
            if random.random() > 0.8: 
                X = self.noise(X)
            if random.random() > 0.5:
                # flip
                X = torchvision.transforms.functional.hflip(X)
                Y = torchvision.transforms.functional.hflip(Y)
                ROI = torchvision.transforms.functional.hflip(ROI)
            if random.random() > 0.5:
                angle = random.randint(-90, 90)
                X = torchvision.transforms.functional.rotate(X, angle)
                Y = torchvision.transforms.functional.rotate(Y, angle)
                ROI = torchvision.transforms.functional.rotate(ROI, angle)
                
            
        Y = (Y > 0.95).float()
        return X.to(torch.float32), Y.to(torch.float32).mean(0), ROI.to(torch.float32).mean(0)


if __name__ == "__main__":
    X, Y, ROI = CustomDataset('/mnt/fastdata/CDNet', '/mnt/fastdata/CDNet', 1, mode='train')[0]
    print(X.shape, Y.shape, ROI.shape)
