from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image


# %%

class MyModel(torch.nn.Module):
    def __init__(self, args):
        config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.drop_path_rate = args.drop_path_rate
        config.classifier_dropout = args.classifier_dropout
        backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", config=config)
        backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        backbone.decode_head.classifier = torch.nn.Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
        super().__init__()
        self.backbone = backbone
        self.texture_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.trans_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=1)
        )

        self.upsampling = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding="same"),
        )        
        
    def forward(self, X):
        X2 = self.backbone(X).logits
        X2 = self.trans_conv(X2)
        texture = self.texture_conv(X[:,:3])
        X2 = torch.cat([X2, texture], dim=1)
        return self.head(X2)

    
