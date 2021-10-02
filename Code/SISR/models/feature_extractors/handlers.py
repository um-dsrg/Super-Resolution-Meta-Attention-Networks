import torch
from SISR.models.feature_extractors import lightCNN, VGGNets


def perceptual_loss_mechanism(name, device=torch.device('cpu'), mode='recognition'):
    if name == 'vgg' and mode == 'p_loss':
        mech = VGGNets.VGGFeatureExtractor(device=device)
    elif name == 'vggface':
        mech = VGGNets.VggFace(mode=mode)
    elif name == 'lightcnn':
        mech = lightCNN.LightCNN_29Layers(device=device)
    else:
        raise Exception('Specified feature extractor not implemented.')
    return mech
