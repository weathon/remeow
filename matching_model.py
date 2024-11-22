# https://github.com/haofeixu/gmflow/blob/main/gmflow/matching.py

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image


# %% 

class MyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=(7, 7), stride=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            torch.nn.Sigmoid()
        )
        
    
    def matching(self, feature1, feature2):
        """
            feature1: torch.Tensor, shape (batch_size, C, 128, 128)
            feature2: torch.Tensor, shape (batch_size, C, 128, 128)
            return torch.Tensor, shape (batch_size, 2, 128, 128)
        """

        feature1_ = feature1[:,1:]#.flatten(2, 3).permute(0, 2, 1)
        feature2_ = feature2[:,1:]#.flatten(2, 3).permute(0, 2, 1)
        assert feature1_.shape == feature2_.shape == (feature1.shape[0], 64*64, feature1.shape[-1]), f"feature1 shape: {feature1_.shape}, feature2 shape: {feature2_.shape}"
        
        corr = torch.bmm(feature1_, feature2_.transpose(1, 2)).softmax(dim=-1)
        grid = torch.stack(torch.meshgrid(torch.arange(64), torch.arange(64)), dim=-1).float().flatten(0, 1).to(corr.device)
        # correspondence = torch.einsum("bll,ld->bld", corr, grid)
        assert grid.shape == (64*64, 2), f"grid shape: {grid.shape}"
        grid = grid[None, ...].repeat(corr.shape[0], 1, 1)
        correspondence = torch.matmul(corr, grid) 
        flow = correspondence - grid
        flow = flow.reshape(feature1.shape[0], 64, 64, 2).permute(0, 3, 1, 2)
        return torch.nn.functional.interpolate(flow, size=(512, 512), mode='bilinear')
        
        
        
        
    def forward(self, X, return_flow=False):
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        flow = self.matching(self.backbone.get_intermediate_layers(current)[0], self.backbone.get_intermediate_layers(long)[0])
        X = torch.cat([current, flow, short], dim=1)
        assert X.shape == (X.shape[0], 8, 512, 512), f"X shape: {X.shape}"
        X = self.decoder(X)
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
    
    




    
