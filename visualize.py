from simple_3dconv import MyModel
from video_dataloader import CustomDataset
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Args:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.05
        self.hidden_dropout_prob = 0.05
        self.drop_path_rate = 0.05
        self.classifier_dropout = 0
        self.ksteps = 200 


model = MyModel(Args())
model.load_state_dict(torch.load("model.pth", weights_only=False))
model = model.cuda()

val_dataset = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 2, "val")

def de_normalize(image):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    image = image * std[None, None, :] + mean[None, None, :]
    return image

def process_and_save_image(index):
    input_image, gt, ROI = val_dataset[index]
    input_image = input_image.unsqueeze(0).cuda()

    with torch.no_grad():
        output_image = (model(input_image).repeat(1, 3, 1, 1) > 0.5) * 1.0
    input_image_np = input_image[:,:3].squeeze().cpu().numpy().transpose(1, 2, 0)
    output_image_np = output_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    input_image_np = de_normalize(input_image_np)
    gt = gt.unsqueeze(0).repeat(3, 1, 1).cpu().numpy().transpose(1, 2, 0)
    print(gt.shape, output_image_np.shape, input_image_np.shape)
    concatenated_image = np.concatenate([input_image_np, gt, output_image_np], axis=1)

    # cv2.imwrite("concatenated_image.png", concatenated_image * 255)
    return concatenated_image * 255


height, width, _ = process_and_save_image(0).shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 1, (width, height))

for i in tqdm(range(0, len(val_dataset), 10)):
    frame = process_and_save_image(i).astype(np.uint8)
    video_writer.write(frame)

video_writer.release()
    