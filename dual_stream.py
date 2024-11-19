from transformers.models.segformer import TwoStreamSegformerEncoder, SegformerConfig, TwoStreamSegformerModel, TwoStreamSegformerForSemanticSegmentation
import torch
import numpy as np
from PIL import Image



class MyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        config = SegformerConfig.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.drop_path_rate = args.drop_path_rate
        config.classifier_dropout = args.classifier_dropout
        self.backbone = TwoStreamSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", config=config)
        self.backbone.decode_head.classifier = torch.nn.Identity()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.1), 
            torch.nn.Conv2d(256, 1, kernel_size=(1 ,1), padding="same"),
            torch.nn.Sigmoid() # That is why same code not training and untill i saw loss higher than 1 i relizewd 
        )
            
        for param in self.backbone.parameters(): 
            param.requires_grad = False
        
        for param in self.backbone.segformer.encoder.BCAs.parameters():
            param.requires_grad = True
    def forward(self, X):
        in_img, long_img, short_img = X[:, :3], X[:, 3:6], X[:, 6:]
        pred = self.backbone(in_img, long_img, short_img).logits 
        pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode='nearest')
        pred = self.proj(pred)
        return pred
    
if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.attention_probs_dropout_prob = 0.1
            self.hidden_dropout_prob = 0.1
            self.drop_path_rate = 0.1
            self.classifier_dropout = 0.1
            self.ksteps = 200 


    model = MyModel(Args())
    model.eval()
    input_tensor = torch.rand(1, 9, 512, 512)
    output = model(input_tensor)
    print(output.shape)
    print(model.backbone.segformer.encoder.BCAs)
    
    
    
