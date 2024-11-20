from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image


# %% 

class MyModel(torch.nn.Module):
    def __init__(self, args):
        config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.drop_path_rate = args.drop_path_rate
        config.classifier_dropout = args.classifier_dropout
        backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512", config=config)
        backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        backbone.decode_head.classifier = torch.nn.Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
        super().__init__()
        self.backbone = backbone
        self.upsampling = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(3, 32, 5, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv3d(32, 64, 5, padding="same"),
            torch.nn.ReLU(),
        )
            
        self.head = torch.nn.Sequential( 
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 64, kernel_size=(7, 7), stride=(1, 1), padding="same"),
            torch.nn.Dropout2d(0.1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding="same"),
            torch.nn.Dropout2d(0.1),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding="same"),
            torch.nn.Sigmoid()
        )        
        
    def forward(self, X):
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2)
        # print(self.temporal_encoder(frames).mean(2).shape)
        X = self.temporal_encoder(frames).mean(2)
        X2 = self.backbone(X).logits
        return self.head(self.upsampling(X2))


    
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
    X = torch.randn(4, 36, 512, 512)
    print(model(X).shape)
