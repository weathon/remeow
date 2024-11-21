from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TimesformerForVideoClassification


# %% 

class MyModel(torch.nn.Module):
    def __init__(self, args):
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.drop_path_rate = args.drop_path_rate
        config.classifier_dropout = args.classifier_dropout
        backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", config=config)
        backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(64 + 9, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        backbone.decode_head.classifier = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        super().__init__()
        self.backbone = backbone
        self.upsampling = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
        self.temporal_encoder = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")
        self.temporal_linear = torch.nn.Linear(8, 1)
        self.dim_linear = torch.nn.Linear(768, 64)
        self.head = torch.nn.Sequential( 
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 32, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            torch.nn.Dropout2d(0.1),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding="same"),
            torch.nn.Sigmoid()
        )        
        
    def encode_video(self, video):
        # video = self.processor(images=video, return_tensors="pt")
        video = torch.nn.functional.interpolate(video, size=(3, 224, 224)) 
        outputs = self.temporal_encoder.timesformer(video).last_hidden_state
        feature = outputs[:,1:].reshape(-1, 8, 14, 14, 768)
        feature = feature.permute(0, 2, 3, 4, 1)
        feature = self.temporal_linear(feature) 
        feature = feature.permute(0, 4, 1, 2, 3).squeeze(1)
        assert feature.shape == (video.shape[0], 14, 14, 768)
        feature = self.dim_linear(feature)
        feature = feature.permute(0, 3, 1, 2)
        feature = torch.nn.functional.interpolate(feature, size=(512, 512), mode='bilinear')
        assert feature.shape == (video.shape[0], 64, 512, 512), feature.shape
        return feature
        
    def forward(self, X):
        frames, long, short = X[:,:-6], X[:,-6:-3], X[:,-3:]
        current = frames[:, :3]
        frames = torch.stack(torch.split(frames, 3, dim=1), dim=2).permute(0, 2, 1, 3, 4)
        X = self.encode_video(frames[:,:8])
        X = torch.nn.functional.interpolate(X, size=(512, 512))
        X = torch.cat([X, long, short, current], dim=1)
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
