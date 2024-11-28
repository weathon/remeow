import torch
import cv2
import os
import numpy as np
import random
import torchvision
from transformers import SegformerForSemanticSegmentation

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

def generate_2d_positional_encoding(height, width, d_model):
    """
    Generate a 2D positional encoding.

    Parameters:
        height (int): The height of the grid.
        width (int): The width of the grid.
        d_model (int): The dimension of the model (should be even).

    Returns:
        numpy.ndarray: A (height x width x d_model) array with positional encodings.
    """
    assert d_model % 2 == 0, "d_model must be even for splitting between x and y."

    # Split the model dimension for x and y encoding
    d_model_half = d_model // 2

    # Create grid of positions
    y_pos, x_pos = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Calculate positional encodings
    def get_encoding(position, d):
        div_term = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pos_enc = np.zeros((position.shape[0], position.shape[1], d))
        pos_enc[..., 0::2] = np.sin(position[..., None] * div_term)
        pos_enc[..., 1::2] = np.cos(position[..., None] * div_term)
        return pos_enc

    x_encoding = get_encoding(x_pos, d_model_half)
    y_encoding = get_encoding(y_pos, d_model_half)

    # Combine x and y encodings
    positional_encoding = np.concatenate([x_encoding, y_encoding], axis=-1)

    return positional_encoding

class MyModel(torch.nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.backbone = backbone
        swin_t = torchvision.models.swin_t()
        self.swin_block = swin_t.features[7][0] #(torch.randn(1, 224, 224, 768))
        self.seg = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.seg.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(18, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.seg.decode_head.classifier = torch.nn.Conv2d(768, 1, kernel_size=(1, 1), stride=(1, 1))
        self.proj = torch.nn.Conv2d(768, 9, kernel_size=(3, 3))
        
    def getflow(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = x.reshape(-1, 30, 30, 768)
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        x = x.permute(0, 2, 3, 1)
        x = self.swin_block(x)
        x = x.permute(0, 3, 1, 2)
        return self.proj(x)
    
    def forward(self, X): 
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        last = frames[:, 3:6]
        current_ = torchvision.transforms.functional.resize(current, (420, 420))
        last_ = torchvision.transforms.functional.resize(last, (420, 420))
        long_ = torchvision.transforms.functional.resize(long, (420, 420))
        
        current_feature = self.backbone.get_intermediate_layers(current_)[0]
        last_feature = self.backbone.get_intermediate_layers(last_)[0]
        # long_feature = self.backbone.get_intermediate_layers(long_)[0]
        
        flow1 = self.getflow(current_feature, last_feature)
        # flow2 = self.getflow(current_feature, long_feature)
        
        flow1 = torch.nn.functional.interpolate(flow1, size=(640, 640), mode='bilinear')
        # flow2 = torch.nn.functional.interpolate(flow2, size=(640, 640), mode='bilinear') should only use one flow, because gmflow failed with empty frame
        new_X = torch.cat([current, flow1, short, long], dim=1)
        seg_map = torch.sigmoid(self.seg(new_X).logits)
        return torch.nn.functional.interpolate(seg_map, size=(640, 640), mode='bilinear')

if __name__ == "__main__":
    model = MyModel(None)
    from video_dataloader import CustomDataset
    import torch
    X, Y, ROI = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "train")[100]
    print(model(X[None]).shape)
    
    # # save model(X[None]) as flow.png
    # flow = model(X[None]) 
    # flow = flow.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    # X = X.squeeze(0)[:3].detach().cpu().numpy().transpose(1, 2, 0)
    # X = cv2.resize(X, (222, 222))
    # cated = np.concatenate([X, flow], axis=1)
    # cv2.imwrite("flow.png", cated * 256)
    
        