
# import os
# from torch.utils.data import Dataset
# import torchvision.transforms.v2 as transforms
# from PIL import Image
# import torch
# import numpy as np
# import random
# import torchvision
# from transformers import AutoImageProcessor
# IMG_SIZE = 512

# cache = {} 


# print_ = lambda x: None
# for i in os.listdir("/home/wg25r/fastdata/CDNet/hist"):
#     path = os.path.join("/home/wg25r/fastdata/CDNet", 'hist', i)
#     print_(path)
#     histgram = torch.load(path, weights_only=False, map_location='cpu')
#     cache[i.replace(".pt", "")] = histgram

# if __name__ == "__main__":
#     print_ = print
# else:
#     print_ = lambda x: None
    
# class CustomDataset(Dataset):
#     def __init__(self, train_path, val_path, args, mode='train'):
#         self.data_path = train_path if mode == 'train' else val_path
#         fold = args.fold
#         self.args = args
#         self.fold = fold
#         self.mode = mode
#         if args.image_size == 512:
#             self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
#         else:
#             self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
#             global IMG_SIZE
#             IMG_SIZE = 640
#         txt_file = os.path.join(val_path, f'{mode}_{fold}.txt')
#         with open(txt_file, 'r') as f:
#             image_names = sorted(f.read().split("\n"))
#         self.mean = torch.tensor([0.485, 0.456, 0.406] * 3)
#         self.std = torch.tensor([0.229, 0.224, 0.225] * 3)
#         if mode == 'train':
#             self.image_names = image_names 
#             self.transform = transforms.Compose([
#                 transforms.ToImage(),
#             ])
#             self.bg_trans = transforms.Compose([
#                  transforms.RandomApply([
#                     transforms.GaussianBlur((3,13), 1), 
#                     ], 0.5),
#                     transforms.RandomApply([
#                         transforms.RandomRotation((-10,10)),
#                     ], 0.5),
#                     transforms.RandomApply([ 
#                         transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.98, 1.02)),
#                     ], 0.5),

#             ])
#         else:
#             random.seed(19890604)
#             self.image_names = sorted(random.sample(image_names, min(512, len(image_names))))
#             # self.image_names = image_names
#             self.transform = transforms.Compose([
#                 transforms.ToImage(),

#             ])
#         self.noise = torchvision.transforms.v2.GaussianNoise(0.1)
#         self.random_resized_crop = transforms.RandomResizedCrop(IMG_SIZE, scale=(0.25, 1.5), ratio=(0.5, 1.5))

#     def crop(self, in_image, long_image, short_image, gt_image, roi_image, histgram):
#         stacked = torch.cat([in_image, long_image, short_image, gt_image, roi_image, histgram], dim=0)
#         cropped = self.random_resized_crop(stacked)
        
#         pointer0 = 0
#         pointer1 = len(in_image)
#         pointer2 = pointer1 + len(long_image)
#         pointer3 = pointer2 + len(short_image)
#         pointer4 = pointer3 + len(gt_image)
#         pointer5 = pointer4 + len(roi_image)
        
#         in_image = cropped[pointer0:pointer1]
#         long_image = cropped[pointer1:pointer2]
#         short_image = cropped[pointer2:pointer3]
#         gt_image = cropped[pointer3:pointer4]
#         roi_image = cropped[pointer4:pointer5]
#         histgram = cropped[pointer5:] 
        
#         return in_image, long_image, short_image, gt_image, roi_image, histgram
    
#     def fake_shadow(self, img):
#         shifted_imgs = torchvision.transforms.functional.affine(img, angle=0, translate=(random.uniform(0, 20), random.uniform(0, 20)), scale=1, shear = random.uniform(-45,45))
#         img = 0.8 * img + 0.2 * shifted_imgs.mean(0, keepdim=True)
#         return img
    
#     def __len__(self):
#         return len(self.image_names)

#     def strong_pan(self, img): 
#         shifteds = [img] 
#         img2 = img
#         for i in range(random.randint(5, 60)):
#             shiftx = random.randint(0, 20)
#             img = torchvision.transforms.functional.affine(img, angle=0, translate=(shiftx, 0), scale=1, shear=0)
#             img2 = torchvision.transforms.functional.affine(img2, angle=0, translate=(-shiftx, 0), scale=1, shear=0)
#             shifteds.append(img) 
#             shifteds.append(img2)

#         return torch.stack(shifteds).to(torch.float).mean(0)
            
#     def weak_pan(self, img):
#         shifteds = [img] 
#         img2 = img
#         for i in range(random.randint(5, 20)):
#             shiftx = random.randint(0, 20)
#             img = torchvision.transforms.functional.affine(img, angle=0, translate=(shiftx, 0), scale=1, shear=0)
#             img2 = torchvision.transforms.functional.affine(img2, angle=0, translate=(-shiftx, 0), scale=1, shear=0)
#             shifteds.append(img) 
#             shifteds.append(img2)

#         return torch.stack(shifteds).to(torch.float).mean(0)
    
#     def close(self, a, b):
#         diff = np.abs(a - b) 
#         return diff < 0.05 * b
    
#     def __getitem__(self, idx):
#         image_name = self.image_names[idx]
#         print_(image_name)
#         video_name = "_".join(image_name.split("_")[:-1])
        
#         long_path = os.path.join(self.data_path, 'long' if self.args.background_type == "mog2" else "sub_long", image_name)
#         short_path = os.path.join(self.data_path, 'short' if self.args.background_type == "mog2" else "sub_short", image_name)
#         gt_path = os.path.join(self.data_path, 'gt', image_name) 

#         long_image = np.array(Image.open(long_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
#         short_image = np.array(Image.open(short_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
#         gt_image = np.array(Image.open(gt_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
#         roi_image = ~self.close(gt_image.mean(-1), 85) * 255

#         long_image, short_image, gt_image, roi_image = self.transform(long_image, short_image, gt_image, roi_image)
#         if self.mode == 'train':
#             if random.random() > 0.9:
#                 print_("Hit 1")
#                 long_image = self.strong_pan(long_image)
#                 short_image = self.weak_pan(short_image) 
#             if random.random() > 0.9:
#                 print_("Hit -1")
#                 long_image = long_image + long_image * (torch.rand(1) * 0.5 - 0.25)
#             if random.random() > 0.9:
#                 print_("Hit -2")
#                 short_image = short_image + short_image * (torch.rand(1) * 0.5 - 0.25)
#             if random.random() > 0.7:
#                 print_("Hit -3")
#                 short_image = self.bg_trans(short_image)
#             if random.random() > 0.7:
#                 print_("Hit -4")
#                 long_image = self.bg_trans(long_image)
#         in_images = []
#         for i in range(0,80,10):
#             image_id = int(image_name.split("_")[-1].split(".")[0].replace("in","")) - i
#             image_id = str(image_id).zfill(6)
#             in_image_path = os.path.join(self.data_path, 'in', "_".join(image_name.split("_")[:-1] + [f"in{image_id}.jpg"]))
#             in_image = np.array(Image.open(in_image_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
#             in_image = self.transform(in_image)
#             in_image = self.image_processor(images=in_image/max(255, in_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
#             in_images.append(in_image)
            
#         histgram = torch.load(os.path.join(self.data_path, 'hist', video_name + ".pt"), weights_only=False, map_location='cpu')
#         histgram = histgram.permute(0, 3, 1, 2).to(torch.float32)
#         histgram = torch.nn.functional.interpolate(histgram, size=(IMG_SIZE, IMG_SIZE), mode="nearest")
#         histgram = histgram.flatten(0, 1)
        
#         in_images = torch.cat(in_images, dim=0)
#         long_image = self.image_processor(images=long_image/max(255, long_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
#         short_image = self.image_processor(images=short_image/max(255, short_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
#         if self.mode == "train":
#             if random.random() > 0.5:
#                 in_image_, long_image_, short_image_, gt_image_, roi_image_, histgram_ = self.crop(in_images, long_image, short_image, gt_image, roi_image, histgram)
#                 assert in_image_.shape == in_images.shape
#                 assert long_image_.shape == long_image.shape
#                 assert short_image_.shape == short_image.shape
#                 assert gt_image_.shape == gt_image.shape
#                 assert roi_image_.shape == roi_image.shape
#                 assert histgram_.shape == histgram.shape
#                 in_images = in_image_
#                 long_image = long_image_
#                 short_image = short_image_
#                 gt_image = gt_image_
#                 roi_image = roi_image_
#                 histgram = histgram_
#             if random.random() > 0.8:
#                 in_images = self.fake_shadow(in_images)
#                 long_image = self.fake_shadow(long_image)
#                 short_image = self.fake_shadow(short_image)
                
#         X = torch.cat([in_images, long_image, short_image], dim=0)
#         if self.mode == "train": 
#             if random.random() > 0.7:
#                 X = X + X * (torch.rand(X.shape[0])[:,None,None] * 0.5 - 0.25)
                
#         ROI = transforms.functional.resize(roi_image, (IMG_SIZE, IMG_SIZE))
#         Y = gt_image
#         Y = transforms.functional.resize(Y, (IMG_SIZE, IMG_SIZE))
#         Y = Y/255
#         ROI = ROI/255
#         ROI[0,0]=1
#         ROI[0,1]=0
#         if self.mode == 'train':
#             if random.random() > 0.8: 
#                 print_("Hit 2")
#                 X = self.noise(X)  
#             if random.random() > 0.5:
#                 # flip
#                 print_("Hit 3")
#                 X = torchvision.transforms.functional.hflip(X)
#                 Y = torchvision.transforms.functional.hflip(Y)
#                 ROI = torchvision.transforms.functional.hflip(ROI)
#             if random.random() > 0.5:
#                 print_("Hit 4")
#                 angle = random.randint(-90, 90)
#                 X = torchvision.transforms.functional.rotate(X, angle)
#                 Y = torchvision.transforms.functional.rotate(Y, angle)
#                 ROI = torchvision.transforms.functional.rotate(ROI, angle)
#                 histgram = torchvision.transforms.functional.rotate(histgram, angle)

#         # assert (histgram_.reshape(51, 3, 512, 512) == histgram).all()
#         if self.args.histogram:
#             X = torch.cat([X, histgram], dim=0) 
#         hard_shadow = self.close(Y, 50/255)
#         Y = (Y > 0.95).float()
#         # if self.args.hard_shadow:
#         #     print_("Hit 5") 
#         #     Y[hard_shadow] = -0.3
#         return X.to(torch.float32), Y.to(torch.float32).mean(0), ROI.to(torch.float32).mean(0)


# if __name__ == "__main__":
#     import argparse
#     import shlex
#     import cv2
#     parser = argparse.ArgumentParser(description="Training script")
#     parser.add_argument('--fold', type=int, required=True, help='Fold number for cross-validation')
#     parser.add_argument('--gpu', type=str, default="0", help='GPU id to use')
#     parser.add_argument('--refine_mode', type=str, default="residual", help='Refine mode', choices=["residual", "direct"])
#     parser.add_argument('--noise_level', type=float, default=1, help='Noise level') 
#     parser.add_argument('--steps', type=int, default=25000, help='Number of steps to train')
#     parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate') 
#     parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
#     parser.add_argument('--mask_upsample', type=str, default="interpolate", help='Mask upsample method', choices=["interpolate", "transpose_conv", "shuffle"])
#     parser.add_argument('--background_type', type=str, default="mog2", help='Background type', choices=["mog2", "sub"])
#     parser.add_argument('--refine_see_bg', action="store_true", help='If refine operator can see background')
#     parser.add_argument('--backbone', type=str, default="4", help='Backbone size to use', choices=["0", "1", "2", "3", "4"])
#     parser.add_argument('--image_size', type=str, default="512", help="Image size", choices=["512", "640"])
#     parser.add_argument('--refine_steps', type=int, default=5, help='Number of refine steps')
#     parser.add_argument('--histogram', action="store_true", help='If use histogram')
#     parser.add_argument('--clip', type=float, default=1, help='Gradient clip norm')
#     parser.add_argument('--note', type=str, default="", help='Note for this run (for logging purpose)')
#     parser.add_argument('--conf_penalty', type=float, default=0, help='Confidence penalty, penalize the model if it is too confident')
#     parser.add_argument('--hard_shadow', action="store_true", help='If use hard shadow')
    
#     argString = '--gpu 0 --fold 2 --noise_level 0.3 --steps 50000 --learning_rate 4e-5 --mask_upsample shuffle --weight_decay 3e-2'
#     args = parser.parse_args(shlex.split(argString))
#     X, Y, ROI = CustomDataset('/mnt/fastdata/CDNet', '/mnt/fastdata/CDNet', args, mode='train')[944]
#     std = torch.tensor([0.5, 0.5, 0.5])
#     mean = torch.tensor([0.5, 0.5, 0.5])
#     X = X[:3]
#     # denorm
#     X = X * std[:, None, None]
#     X = X + mean[:, None, None]
#     cv2.imwrite("test.png", X.numpy().transpose(1, 2, 0) * 255)
#     print_(Y.shape)
#     Y = Y.numpy()
#     # Y = (Y - Y.min()) / (Y.max() - Y.min())
#     # cv2.imwrite("test_gt.png", Y * 255)
#     import pylab
#     pylab.imshow(Y)
#     pylab.show()
#     pylab.savefig("test_gt.png")
#     print_(X.shape, Y.shape, ROI.shape)

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

# cache = {} 

# for i in os.listdir("/home/wg25r/fastdata/CDNet/hist"):
#     path = os.path.join("/home/wg25r/fastdata/CDNet", 'hist', i)
#     print_(path)
#     histgram = torch.load(path, weights_only=False, map_location='cpu')
#     cache[i.replace(".pt", "")] = histgram
NEAREST = torchvision.transforms.InterpolationMode.NEAREST
if __name__ == "__main__":
    print_ = print
else:
    print_ = lambda x: None
    
class CustomDataset(Dataset):
    def __init__(self, train_path, val_path, args, mode='train', filename=False):
        self.filename = filename
        self.data_path = train_path if mode == 'train' else val_path
        fold = args.fold
        self.args = args
        self.fold = fold
        self.mode = mode
        if args.image_size == 512:
            self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        else:
            self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            global IMG_SIZE
            IMG_SIZE = 640
        txt_file = os.path.join(val_path, f'{mode}_{fold}.txt')
        with open(txt_file, 'r') as f:
            image_names = sorted(f.read().split("\n"))
        self.mean = torch.tensor([0.485, 0.456, 0.406] * 3)
        self.std = torch.tensor([0.229, 0.224, 0.225] * 3)
        if mode == 'train':
            self.image_names = []
            for image_name in image_names:
                video_name = "_".join(image_name.split("_")[:2])
                cat_name =  image_name.split("_")[0]
                if cat_name in ["PTZ", "nightVideos", "lowFramerate", "turbulence"]:
                    self.image_names.extend([image_name] * 10)
                else:
                    self.image_names.append(image_name)
                    
            
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
            random.seed(19890604)
            self.image_names = sorted(random.sample(image_names, min(self.args.val_size, len(image_names))))
            # self.image_names = image_names
            self.transform = transforms.Compose([
                transforms.ToImage(),

            ])
        self.noise = torchvision.transforms.v2.GaussianNoise(0.1)
        self.random_resized_crop = transforms.RandomResizedCrop(IMG_SIZE, scale=(0.25, 1.5), ratio=(0.5, 1.5), interpolation=NEAREST)

    def crop(self, in_image, long_image, short_image, gt_image, roi_image):
        stacked = torch.cat([in_image, long_image, short_image, gt_image, roi_image], dim=0)
        cropped = self.random_resized_crop(stacked)
        
        pointer0 = 0
        pointer1 = len(in_image)
        pointer2 = pointer1 + len(long_image)
        pointer3 = pointer2 + len(short_image)
        pointer4 = pointer3 + len(gt_image)
        pointer5 = pointer4 + len(roi_image)
        
        in_image = cropped[pointer0:pointer1]
        long_image = cropped[pointer1:pointer2]
        short_image = cropped[pointer2:pointer3]
        gt_image = cropped[pointer3:pointer4]
        roi_image = cropped[pointer4:pointer5]
        
        return in_image, long_image, short_image, gt_image, roi_image
    
    def fake_shadow(self, img):
        shifted_imgs = torchvision.transforms.functional.affine(img, angle=0, translate=(random.uniform(0, 20), random.uniform(0, 20)), scale=1, shear = random.uniform(-45,45))
        img = 0.8 * img + 0.2 * shifted_imgs.mean(0, keepdim=True)
        return img
    
    def __len__(self): 
        return len(self.image_names)

    def strong_pan(self, img): 
        shifteds = [img] 
        img2 = img
        for i in range(random.randint(5, 60)):
            shiftx = random.randint(10, 20)
            img = torchvision.transforms.functional.affine(img, angle=random.random()*2, translate=(shiftx, 0), scale=1, shear=0)
            img2 = torchvision.transforms.functional.affine(img2, angle=random.random()*2, translate=(-shiftx, 0), scale=1, shear=0)
            shifteds.append(img.clone()) 
            shifteds.append(img2.clone())
        # tmp = torch.stack(shifteds).to(torch.float).mean(0)
        # tmp = tmp.detach().cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite("pan.png", tmp)
        return torch.stack(shifteds).to(torch.float).mean(0)
            
    def weak_pan(self, img):
        shifteds = [img] 
        img2 = img
        for i in range(random.randint(5, 20)):
            shiftx = random.randint(10, 20)
            img = torchvision.transforms.functional.affine(img, angle=random.random()*2, translate=(shiftx, 0), scale=1, shear=0)
            img2 = torchvision.transforms.functional.affine(img2, angle=random.random()*2, translate=(-shiftx, 0), scale=1, shear=0)
            shifteds.append(img.clone()) 
            shifteds.append(img2.clone())
        # print(len(shifteds))
        return torch.stack(shifteds).to(torch.float).mean(0)
    
    def close(self, a, b, threshold=0.02 * 255):
        diff = np.abs(a - b) 
        return diff < threshold
    
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        print_(image_name)
        video_name = "_".join(image_name.split("_")[:-1])
        
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
            # if 1:
            # if 0:
                print_("Hit 1")
                long_image = self.strong_pan(long_image)
                short_image = self.weak_pan(short_image) 
            if random.random() > 0.9:
                print_("Hit -1")
                long_image = long_image + long_image * (torch.rand(1) * 0.5 - 0.25)
            if random.random() > 0.9:
                print_("Hit -2")
                short_image = short_image + short_image * (torch.rand(1) * 0.5 - 0.25)
            if random.random() > 0.7:
                print_("Hit -3")
                short_image = self.bg_trans(short_image)
            if random.random() > 0.7:
                print_("Hit -4")
                long_image = self.bg_trans(long_image)

                
        in_images = []
        for i in range(0,80,10):
            image_id = int(image_name.split("_")[-1].split(".")[0].replace("in","")) - i
            image_id = str(image_id).zfill(6)
            in_image_path = os.path.join(self.data_path, 'in', "_".join(image_name.split("_")[:-1] + [f"in{image_id}.jpg"]))
            if self.args.recent_frames != "none" or i==0:
                in_image = np.array(Image.open(in_image_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
                if random.random() > 0.7:
                    x1, y1 = random.randint(0, 512-200), random.randint(0, 512-200)
                    x2, y2 = x1 + random.randint(100, 200), y1 + random.randint(100, 200)
                    in_image[y1:y2, x1:x2] = 255 - in_image[y1:y2, x1:x2]
                    
                in_image = self.transform(in_image)
                in_image = self.image_processor(images=in_image/max(255, in_image.max()), return_tensors='pt', do_rescale=False)['pixel_values'][0]
                in_image = in_image + in_image * (torch.rand(1) * 0.5 - 0.25)
                
            in_images.append(in_image)
            
        # histgram = torch.load(os.path.join(self.data_path, 'hist', video_name + ".pt"), weights_only=False, map_location='cpu')
        # histgram = histgram.permute(0, 3, 1, 2).to(torch.float32)
        # histgram = torch.nn.functional.interpolate(histgram, size=(IMG_SIZE, IMG_SIZE), mode="nearest")
        # histgram = histgram.flatten(0, 1)
        
        in_images = torch.cat(in_images, dim=0)
        long_image[long_image<0] = 0
        short_image[short_image<0] = 0
        long_image[long_image>255] = 255
        short_image[short_image>255] = 255
        
        long_image = self.image_processor(images=long_image/255, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        short_image = self.image_processor(images=short_image/255, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        if self.mode == "train":
            if random.random() > 0.5:
                in_image_, long_image_, short_image_, gt_image_, roi_image_ = self.crop(in_images, long_image, short_image, gt_image, roi_image)
                assert in_image_.shape == in_images.shape
                assert long_image_.shape == long_image.shape
                assert short_image_.shape == short_image.shape
                assert gt_image_.shape == gt_image.shape
                assert roi_image_.shape == roi_image.shape
                in_images = in_image_
                long_image = long_image_
                short_image = short_image_
                gt_image = gt_image_
                roi_image = roi_image_
            # if random.random() > 0.8:
            #     in_images = self.fake_shadow(in_images)
            #     long_image = self.fake_shadow(long_image)
            #     short_image = self.fake_shadow(short_image)
        # if self.args.use_difference:
        #     long_image = long_image - in_images[0]
        #     short_image = short_image - in_images[0] 
        #  this is the reason reverse and no pan on by default. check the main file now 
            
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
                print_("Hit 2")
                X = self.noise(X)  
            if random.random() > 0.5:
                # flip
                print_("Hit 3")
                X = torchvision.transforms.functional.hflip(X)
                Y = torchvision.transforms.functional.hflip(Y)
                ROI = torchvision.transforms.functional.hflip(ROI)
            if random.random() > 0.5:
                print_("Hit 4")
                angle = random.randint(-90, 90)
                X = torchvision.transforms.functional.rotate(X, angle)
                Y = torchvision.transforms.functional.rotate(Y, angle)
                ROI = torchvision.transforms.functional.rotate(ROI, angle, interpolation=NEAREST)

        # assert (histgram_.reshape(51, 3, 512, 512) == histgram).all()
 
        hard_shadow = self.close(Y, 50/255, threshold=0.02)
        Y = (Y > 0.95).float()
        if self.args.num_classes == 3:
            print_("Hit 5")
            Y[hard_shadow] = 2
        if self.filename:
            return X.to(torch.float32), Y.to(torch.float32).mean(0), ROI.to(torch.float32).mean(0), image_name
        else:
            return X.to(torch.float32), Y.to(torch.float32).mean(0), ROI.to(torch.float32).mean(0)


if __name__ == "__main__":
    import argparse
    import shlex
    import cv2
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--fold', type=int, required=True, help='Fold number for cross-validation')
    parser.add_argument('--gpu', type=str, default="0", help='GPU id to use')
    parser.add_argument('--refine_mode', type=str, default="residual", help='Refine mode', choices=["residual", "direct"])
    parser.add_argument('--noise_level', type=float, default=1, help='Noise level') 
    parser.add_argument('--steps', type=int, default=25000, help='Number of steps to train')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate') 
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--mask_upsample', type=str, default="interpolate", help='Mask upsample method', choices=["interpolate", "transpose_conv", "shuffle"])
    parser.add_argument('--background_type', type=str, default="mog2", help='Background type', choices=["mog2", "sub"])
    parser.add_argument('--refine_see_bg', action="store_true", help='If refine operator can see background')
    parser.add_argument('--backbone', type=str, default="4", help='Backbone size to use', choices=["0", "1", "2", "3", "4"])
    parser.add_argument('--image_size', type=str, default="512", help="Image size", choices=["512", "640"])
    parser.add_argument('--refine_steps', type=int, default=5, help='Number of refine steps')
    parser.add_argument('--histogram', action="store_true", help='If use histogram')
    parser.add_argument('--clip', type=float, default=1, help='Gradient clip norm')
    parser.add_argument('--note', type=str, default="", help='Note for this run (for logging purpose)')
    parser.add_argument('--conf_penalty', type=float, default=0, help='Confidence penalty, penalize the model if it is too confident')
    parser.add_argument('--hard_shadow', action="store_true", help='If use hard shadow')
    parser.add_argument('--use_difference', action="store_true", help='If use difference between current and background ratehr than background frame')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--recent_frames', type=str, default="conv3d", help='Recent frames method', choices=["conv3d", "linear", "none"])
    parser.add_argument('--checkpoint', type=str, default="", help='Load checkpoint')

    argString = '--gpu 0 --fold 2 --noise_level 0.3 --steps 50000 --learning_rate 4e-5 --mask_upsample shuffle --weight_decay 3e-2 --hard_shadow'
    args = parser.parse_args(shlex.split(argString))
    X, Y, ROI = CustomDataset('/mnt/fastdata/CDNet', '/mnt/fastdata/CDNet', args, mode='train')[9440]
    #random.randint(0, 1000)
    std = torch.tensor([0.5, 0.5, 0.5])
    mean = torch.tensor([0.5, 0.5, 0.5])
    bg = X[-6:-3]
    X = X[:3]
    print(bg.shape)
    # denorm
    X = X * std[:, None, None]
    X = X + mean[:, None, None]
    
    bg = bg * std[:, None, None]
    bg = bg + mean[:, None, None]
    cv2.imwrite("test.png", X.numpy().transpose(1, 2, 0) * 255)
    print_(Y.shape)
    Y = Y.numpy()
    # Y = (Y - Y.min()) / (Y.max() - Y.min())
    # cv2.imwrite("test_gt.png", Y * 255)
    import pylab
    pylab.subplot(1, 3, 1)
    pylab.imshow(Y, interpolation="nearest", origin='upper', vmax=2, vmin=0)
    pylab.colorbar() 
    pylab.subplot(1, 3, 2)
    pylab.imshow(X[:3].permute(1, 2, 0), interpolation="nearest", origin='upper')
    pylab.subplot(1, 3, 3)
    pylab.imshow(bg.permute(1, 2, 0), interpolation="nearest", origin='upper')
    pylab.savefig("test_gt.png")
    print_(X.shape, Y.shape, ROI.shape)
