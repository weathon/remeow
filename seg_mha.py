from transformers import SegformerConfig, SegformerForSemanticSegmentation
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
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", config=config)
        self.backbone.decode_head.classifier = torch.nn.Identity()
        self.local_window = torch.nn.Conv2d(256, 64, kernel_size=(9, 9), padding="same")
        self.bca = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, dropout=0.3, batch_first=True), 1
        )
        
        self.proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.1), 
            torch.nn.Conv2d(64, 1, kernel_size=(1 ,1), padding="same"),
            torch.nn.Sigmoid() # That is why same code not training and untill i saw loss higher than 1 i relizewd 
        )
        
        # for param in self.backbone.parameters(): 
        #     param.requires_grad = False 
             
            
    
    def forward(self, X):
        in_img, long_img, short_img = X[:, :3], X[:, 3:6], X[:, 6:]
        in_pred = self.backbone(in_img).logits 
        long_pred = self.backbone(long_img).logits
        short_pred = self.backbone(short_img).logits
        
        in_pred = self.local_window(in_pred)
        long_pred = self.local_window(long_pred)
        short_pred = self.local_window(short_pred) 
        # target dim = (B * H * W, 3, DIM)
        # Current dim = (B, DIM, H, W)
        B, DIM, H, W = in_pred.shape
        seq = torch.stack([in_pred, long_pred, short_pred], dim=0)
        assert seq.shape == (3, B, DIM, H, W)
        seq = seq.flatten(3,4)
        assert seq.shape == (3, B, DIM, H*W)
        seq = seq.permute(1, 3, 0, 2)
        assert seq.shape == (B, H*W, 3, DIM)
        seq = seq.flatten(0,1)
        assert seq.shape == (B*H*W, 3, DIM)
        seq = self.bca(seq)
        assert seq.shape == (B*H*W, 3, DIM)
        seq = seq.reshape(B, H*W, 3, DIM)
        assert seq.shape == (B, H*W, 3, DIM)
        seq = seq.permute(0, 2, 3, 1)
        assert seq.shape == (B, 3, DIM, H*W)
        seq = seq.reshape(B, 3, DIM, H, W)
        assert seq.shape == (B, 3, DIM, H, W)
        seq = seq[:,0]
        assert seq.shape == (B, DIM, H, W)
        
        

        pred = torch.nn.functional.interpolate(seq, size=(512, 512), mode='nearest')
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
    
    
    
