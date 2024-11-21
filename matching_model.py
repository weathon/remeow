# https://github.com/haofeixu/gmflow/blob/main/gmflow/matching.py

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image


# %% 

class MyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        config = SegformerConfig.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.drop_path_rate = args.drop_path_rate
        config.classifier_dropout = args.classifier_dropout
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512", config=config)
        self.backbone.decode_head.classifier = torch.nn.Identity()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(7, 32, kernel_size=(7, 7), stride=(1, 1), padding="same"),
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
        feature1 = torch.nn.functional.interpolate(feature1, size=(64, 64), mode='bilinear')
        feature2 = torch.nn.functional.interpolate(feature2, size=(64, 64), mode='bilinear')
        feature1_ = feature1.flatten(2, 3).permute(0, 2, 1)
        feature2_ = feature2.flatten(2, 3).permute(0, 2, 1)
        assert feature1_.shape == feature2_.shape == (feature1.shape[0], 64*64, feature1.shape[1]), f"feature1 shape: {feature1_.shape}, feature2 shape: {feature2_.shape}"
        
        corr = torch.bmm(feature1_, feature2_.transpose(1, 2)).softmax(dim=-1)
        grid = torch.stack(torch.meshgrid(torch.arange(64), torch.arange(64)), dim=-1).float().flatten(0, 1).to(corr.device)
        correspondence = torch.einsum("bik,kj->bij", corr, grid)
        flow = correspondence - grid
        flow = flow.reshape(feature1.shape[0], 64, 64, 2).permute(0, 3, 1, 2)
        return torch.nn.functional.interpolate(flow, size=(512, 512), mode='bilinear')
        
        
        
        
    def forward(self, X):
        current = X[:, :3]
        long = X[:, -6:-3]
        short = X[:, -3:]
        current_feature = self.backbone(current).logits
        long_feature = self.backbone(long).logits
        short_feature = self.backbone(short).logits
        
        flow1 = self.matching(current_feature, long_feature)
        flow2 = self.matching(current_feature, short_feature)
        image = torch.cat([flow1, flow2, current], dim=1)
        return self.decoder(image)
    
if __name__ == "__main__":
    # %%
    class Args:
        def __init__(self):
            self.attention_probs_dropout_prob = 0.05
            self.hidden_dropout_prob = 0.05
            self.drop_path_rate = 0.05
            self.classifier_dropout = 0
            self.ksteps = 200 


    args = Args()
    model = MyModel(args)
    X = torch.randn(4, 9, 512, 512)
    print(model(X).shape)




    
