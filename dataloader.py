import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
import torch

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
                transforms.RandomResizedCrop((512, 512)),
            ])
        else:
            self.image_names = image_names
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8),
                transforms.Resize((512, 512)),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        long_path = os.path.join(self.data_path, 'long', image_name)
        short_path = os.path.join(self.data_path, 'short', image_name)
        roi_path = os.path.join(self.data_path, 'ROI', image_name)
        gt_path = os.path.join(self.data_path, 'gt', image_name) 
        in_path = os.path.join(self.data_path, 'in', image_name)

        long_image = Image.open(long_path).convert('RGB').resize((512, 512))
        short_image = Image.open(short_path).convert('RGB').resize((512, 512))
        roi_image = Image.open(roi_path).convert('RGB').resize((512, 512))
        gt_image = Image.open(gt_path).convert('RGB').resize((512, 512))
        in_image = Image.open(in_path).convert('RGB').resize((512, 512))

        long_image, short_image, roi_image, gt_image, in_image = self.transform(long_image, short_image, roi_image, gt_image, in_image)

        return {
            'long': long_image,
            'short': short_image,
            'ROI': roi_image,
            'gt': gt_image
        }
    

# test

if __name__ == '__main__':
    import pylab
    train_path = '/mnt/fastdata/preaug_cdnet/'
    val_path = '/mnt/fastdata/CDNet/'
    fold = 1
    dataset = CustomDataset(train_path, val_path, fold, mode='train')
    pylab.figure(figsize=(20, 20))
    pylab.subplot(1, 4, 1)
    pylab.imshow(dataset[0]['long'].permute(1, 2, 0))
    pylab.subplot(1, 4, 2)
    pylab.imshow(dataset[0]['short'].permute(1, 2, 0))
    pylab.subplot(1, 4, 3)
    pylab.imshow(dataset[0]['ROI'].permute(1, 2, 0))
    pylab.subplot(1, 4, 4)
    pylab.imshow(dataset[0]['gt'].permute(1, 2, 0))
    pylab.savefig('test.png')
