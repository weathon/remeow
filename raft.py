# https://github.com/haofeixu/gmflow/blob/main/gmflow/matching.py

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image
import torchvision


# %% 

class MyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.decoder = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.decoder.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(11, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.decoder.decode_head.classifier = torch.nn.Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1))
        self.raft = torchvision.models.optical_flow.raft_large(pretrained=True)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            torch.nn.Sigmoid()
        )
        

        
        
        
    def forward(self, X, return_flow=False):
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        last = frames[:, 3:6]
        flow = self.raft(current, last)[-1]
        X = torch.cat([current, flow, long, short], dim=1)
        assert X.shape == (X.shape[0], 11, 512, 512), f"X shape: {X.shape}"
        X = self.decoder(X)
        X = self.head(X.logits)
        X = torch.nn.functional.interpolate(X, size=(512, 512), mode='bilinear')
        if return_flow:
            return X, flow
        return X
if __name__ == "__main__":
    # %%
    from video_dataloader import CustomDataset
    import cv2
    class Args:
        def __init__(self):
            self.attention_probs_dropout_prob = 0.05
            self.hidden_dropout_prob = 0.05
            self.drop_path_rate = 0.05
            self.classifier_dropout = 0
            self.ksteps = 200 


    args = Args()
    model = MyModel(args)
    X = CustomDataset("/mnt/fastdata/CDNet", "/mnt/fastdata/CDNet", 3, "val")[0][0].unsqueeze(0)
    pred = model(X, return_flow=True)
    print(pred[1][0][0].shape)
    cv2.imwrite('test.png', pred[1][0][1].detach().cpu().numpy()*255)
    cv2.imwrite("source.png", X[0, :3].permute(1, 2, 0).detach().cpu().numpy()*255)
    
    




    
