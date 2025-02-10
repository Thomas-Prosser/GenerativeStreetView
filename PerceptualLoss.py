import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19 model from torchvision
        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        
        # Select layers for perceptual loss computation
        self.selected_layers = {'input': 0, 'conv1_2': 3, 'conv2_2': 8, 'conv3_2': 13, 'conv4_2': 22, 'conv5_2': 31}
        
        # Extract only required layers
        self.model = nn.Sequential(*list(vgg.children())[:max(self.selected_layers.values()) + 1])
        self.model.to(device)
        
        # Disable gradient updates for VGG19
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Define layer weights for loss computation
        self.weights = {'input': 1.0, 'conv1_2': 1/2.6, 'conv2_2': 1/4.8, 'conv3_2': 1/3.7, 'conv4_2': 1/5.6, 'conv5_2': 10/1.5}
        self.device = device
    
    def forward(self, real_img, fake_img):
        """
        Compute perceptual loss between real_img and fake_img.
        Images should be normalized in the range [-1, 1].
        """
        # Scale images from [-1,1] to [0,255]
        real_img = (real_img + 1.0) / 2.0 * 255.0
        fake_img = (fake_img + 1.0) / 2.0 * 255.0
        
        # Extract features from images
        features_real = {}
        features_fake = {}
        x_real, x_fake = real_img, fake_img
        
        for name, layer in self.model._modules.items():
            x_real = layer(x_real)
            x_fake = layer(x_fake)
            
            if int(name) in self.selected_layers.values():
                layer_name = [k for k, v in self.selected_layers.items() if v == int(name)][0]
                features_real[layer_name] = x_real
                features_fake[layer_name] = x_fake
        
        # Compute perceptual loss
        loss = 0.0
        for layer_name in self.selected_layers.keys():
            loss += self.weights[layer_name] * F.l1_loss(features_real[layer_name], features_fake[layer_name])
        
        return loss

if __name__ == "__main__":
    print(1)