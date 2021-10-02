from torch import nn
import torch
from SISR.models.feature_extractors.handlers import perceptual_loss_mechanism


class PerceptualMechanism(nn.Module):
    def __init__(self, device=torch.device('cpu'), lambda_pixel=1, lambda_per=0.01):
        super(PerceptualMechanism, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_per = lambda_per
        self.vgg_extractor = perceptual_loss_mechanism('vgg', mode='p_loss', device=device)
        self.vgg_extractor.to(device)
        self.vgg_extractor.eval()
        self.vgg_loss = nn.L1Loss()
        self.pixel_loss = nn.L1Loss()
        self.device = device

    def forward(self, sr, y):
        gen_features = self.vgg_extractor(sr)
        real_features = self.vgg_extractor(y).detach()
        vgg_loss = self.vgg_loss(gen_features, real_features)
        return (self.lambda_pixel*self.pixel_loss(sr, y)) + (self.lambda_per*vgg_loss)
