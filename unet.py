import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=768*2, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (Upsampling)
        self.up1 = self.up_conv(1024, 512)
        self.dec1 = self.conv_block(1024, 512)  # Skip connection
        
        self.up2 = self.up_conv(512, 256)
        self.dec2 = self.conv_block(512, 256)  # Skip connection
        
        self.up3 = self.up_conv(256, 128)
        self.dec3 = self.conv_block(256, 128)  # Skip connection
        
        self.up4 = self.up_conv(128, 64)
        self.dec4 = self.conv_block(128, 64)  # Skip connection
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Dropout2d(0.05),
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder path
        d1 = self.up1(b)
        d1 = torch.cat((d1, e4), dim=1)  # Skip connection
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat((d2, e3), dim=1)  # Skip connection
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat((d3, e2), dim=1)  # Skip connection
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat((d4, e1), dim=1)  # Skip connection
        d4 = self.dec4(d4)
        
        # Final Convolution
        out = self.final_conv(d4)
        return out

if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=768*2, out_channels=1)
    x = torch.randn(1, 768*2, 512, 512)  # Example input
    output = model(x)
    print(output.shape)  # Should be [1, 1, 256, 256]
