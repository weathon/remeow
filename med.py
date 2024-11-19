# %%
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, AutoImageProcessor, BeitForSemanticSegmentation
# backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# backbone.decode_head.classifier = torch.nn.Identity()
# import sys
image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
backbone = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
backbone.auxiliary_head.classifier = torch.nn.Identity()
backbone.classifier = torch.nn.Identity()
backbone.decode_head.classifier = torch.nn.Identity()
# sys.path.append("./DIS/IS-Net")
# from models import *
# backbone = ISNetDIS().cuda()
# backbone.load_state_dict(torch.load("/home/wg25r/isnet-general-use.pth"))

# %%
images=image_processor(images=torch.rand(1, 3, 224, 224), return_tensors="pt")
backbone(**images).logits.shape

# %%
# print(backbone(torch.randn(1, 3, 512, 512).cuda())[1][0].shape)

def model_fn(x):
    X = image_processor(images=x, return_tensors="pt").pixel_values.cuda()
    pred = backbone(pixel_values=X).logits
    return torch.nn.functional.interpolate(pred, size=(320, 320), mode="bilinear", align_corners=False)

# %%
backbone.eval()
import os

# %%
import torch
frames = os.listdir('/mnt/fastdata/dataset/turbulence/turbulence0/input')[::5]
import tqdm
features = torch.zeros((len(frames), 768, 320, 320))
len(frames)

# %%
from PIL import Image
from transformers import AutoImageProcessor
# image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
backbone = backbone.cuda()

# %%
import torchvision
import torchvision.transforms.v2
import numpy as np
for i in tqdm.tqdm(range(len(frames))):
    with torch.no_grad():
        frame = frames[i]
        img = Image.open(f'/mnt/fastdata/dataset/turbulence/turbulence0/input/{frame}').resize((320, 320))
        res = model_fn(img).cpu()
        features[i] = res   

# %%
# mean = features[0]
mean = torch.mean(features, dim=0)

sim = torch.nn.CosineSimilarity(dim=1)(mean, features)


# %%
difference = 1 - sim

# %%
difference.shape

# %%
difference = difference.cpu().numpy()
# difference = difference.mean(1)


# %%
# save difference into a mp4 video 
import cv2
import numpy as np
import os 
import tqdm

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('difference.mp4', fourcc, 10, (320 * 2, 320))
for i in tqdm.tqdm(range(len(difference))):
    mask = (difference[i]) * 255 
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    raw = cv2.imread(f'/mnt/fastdata/dataset/turbulence/turbulence0/input/{frames[i]}')
    raw = cv2.resize(raw, (320, 320))
    frame = np.concatenate((raw, mask), axis=1)
    video.write(frame)
video.release() 

# %%
# kmeans (kind of similar to mog) then chose the most common cluster as the background (even gpt knows this)


