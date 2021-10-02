import numpy as np
import torch
import PIL.Image
import random


def landmark_crop(image, crop_size, landmarks):
    if type(landmarks) == str:
        centroid = (image.width/2, image.height/2)
    else:
        centroid = (landmarks.max(0) + landmarks.min(0))/2
    l_pos, t_pos = centroid[0] - (crop_size[0]/2), centroid[1] - (crop_size[1]/2)
    cropped_image = image.crop((l_pos, t_pos, l_pos + crop_size[0], t_pos + crop_size[1]))

    if type(landmarks) == str:
        scaled_landmarks = landmarks
    else:
        scaled_landmarks = np.copy(landmarks)
        scaled_landmarks[:, 0] = landmarks[:, 0] - l_pos
        scaled_landmarks[:, 1] = landmarks[:, 1] - t_pos

    return cropped_image, scaled_landmarks


def detect_negative_landmarks(landmarks):
    if (landmarks < 0).any():
        return True
    else:
        return False


def downsample(image, scale, jm=False):
    """
    Downsamples image appropriately, according to scale factor.
    :param image: Input HR image.
    :param scale: Scale factor for downsampling.
    :param jm: Set to True if special consideration for JM compression needs to be made.
    :return: Cropped HR image, downsampled LR image.
    """
    if jm:
        corrected_width = ((image.width // scale) // 2) * 2  # JM only accepts even dimensions (for unknown reasons)
        corrected_height = ((image.height // scale) // 2) * 2
    else:
        corrected_width = image.width // scale
        corrected_height = image.height // scale

    r_width = corrected_width * scale  # re-scaling HR image to dimensions which fit into the selected scale
    r_height = corrected_height * scale

    r_image = CenterCrop(width=r_width, height=r_height)(image)

    l_image = r_image.resize((r_width // scale, r_height // scale), resample=PIL.Image.BICUBIC)  # downsize
    return r_image, l_image


def rgb_to_ycbcr(img, y_only=True, max_val=1, im_type='png'):  # image always expected in C, H, W format
    """
    Converts RGB image to YCbCr.
    :param img: CxHxW Image.
    :param y_only: Set as true to only retrieve luminance values.
    :param max_val: Image dynamic range.
    :param im_type: PNG or JPG format.
    :return:  Converted Image.
    """
    if im_type == 'jpg':

        bias_c = 128.*(max_val/255)

        y = (0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :])

        if y_only:
            return y, None, None

        cb = bias_c + (-0.168736 * img[0, :, :] - 0.331264 * img[1, :, :] + 0.5 * img[2, :, :])
        cr = bias_c + (0.5 * img[0, :, :] - 0.418688 * img[1, :, :] - 0.081312 * img[2, :, :])

    else:
        bias_y = 16.*(max_val/255)
        bias_c = 128.*(max_val/255)

        y = bias_y + (65.481 * img[0, :, :] + 128.553 * img[1, :, :] + 24.966 * img[2, :, :]) / 255.

        if y_only:
            return y, None, None

        cb = bias_c + (-37.797 * img[0, :, :] - 74.203 * img[1, :, :] + 112.0 * img[2, :, :]) / 255.
        cr = bias_c + (112.0 * img[0, :, :] - 93.786 * img[1, :, :] - 18.214 * img[2, :, :]) / 255.

    return y, cb, cr


def ycbcr_to_rgb(img, max_val=1, im_type='png'):  # image always expected in C, H, W format
    """
    Converts YCbCr image to RGB.
    :param img: CxHxW Image.
    :param max_val: Image dynamic range.
    :param im_type: PNG or JPG format.
    :return:  Converted Image.
    """
    if im_type == 'jpg':
        bias = 128.*(max_val/255)

        r = img[0, :, :] + 1.402 * img[2, :, :] - 1.402 * bias
        g = img[0, :, :] - 0.344136 * img[1, :, :] - 0.714136 * img[2, :, :] + (0.714136 + 0.344136) * bias
        b = img[0, :, :] + 1.772 * img[1, :, :] - 1.772 * bias

    else:
        bias_r = 222.921*(max_val/255)
        bias_g = 135.576*(max_val/255)
        bias_b = 276.836*(max_val/255)

        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - bias_r
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + bias_g
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - bias_b

    return r, g, b


# TODO: also accept H,W,C images
# TODO: find way to perform conversion using matrix multiplication for faster action
def ycbcr_convert(img, y_only=True, max_val=1, im_type='png', input='rgb'):
    """
    RGB to YCbCr converter using ITU-R BT.601 format (https://en.wikipedia.org/wiki/YCbCr).
    Can perform forward and inverse operation across different modes.
    :param img: Image to convert.  Must be in C, H, W format (channels, height, width)
    :param y_only: Select whether to output luminance channel only.
    :param max_val: Image maximum pixel value.
    :param im_type: Specify image type - different conversion performed for jpg images.
    :param input: Specify whether the input is in rgb or ycbcr format.
    :return: Transformed image.
    """

    if type(img) == np.ndarray:
        form = 'numpy'
    elif type(img) == torch.Tensor:
        form = 'torch'
    else:
        raise Exception('Unknown Type', type(img))

    if len(img.shape) == 4:
        img = img.squeeze(0)

    if input == 'ycbcr':
        a, b, c = ycbcr_to_rgb(img, max_val=max_val, im_type=im_type)
    elif input == 'rgb':
        a, b, c = rgb_to_ycbcr(img, max_val=max_val, y_only=y_only, im_type=im_type)

    if form == 'numpy':
        if y_only and input == 'rgb':
            return np.expand_dims(a, axis=0)
        else:
            return np.array([a, b, c])
    elif form == 'torch':
        if y_only and input == 'rgb':
            return torch.unsqueeze(a, 0)
        else:
            return torch.stack([a, b, c], 0)


def scale_and_luminance_crop(im, max_val=1, target_max=255):

    if type(im) == np.ndarray:
        im_np = np.copy(im)
    elif type(im) == torch.Tensor:
        im_np = im.numpy()
    else:
        raise Exception('Unknown Type', type(im))

    im_rgb = ycbcr_convert(im_np, input='ycbcr', max_val=max_val)
    im_rgb *= target_max/max_val
    im_rgb = np.clip(im_rgb, 0, target_max)
    im_ycbcr = ycbcr_convert(im_rgb, input='rgb', max_val=target_max, y_only=False)

    return im_ycbcr, im_rgb


class RGBtoYCbCrConverter:

    def __init__(self, im_type='jpg', y_only=True, max_val=1):
        """
        Class used by Pytorch Data handler to convert RGB images.
        :param im_type: PNG or JPG.
        :param y_only: Set as true to only retrieve luminance values.
        :param max_val: Image dynamic range.
        """
        self.im_type = im_type
        self.y_only = y_only
        self.max_val = max_val

    def __call__(self, image):
        return ycbcr_convert(image, y_only=self.y_only, max_val=self.max_val, im_type=self.im_type, input='rgb')

    def __repr__(self):
        return self.__class__.__name__ + '()'


def center_crop(image, height, width):
    """
    Base center cropping function
    :param image: Input image.
    :param height: Image crop height.
    :param width: Image crop width.
    :return:
    """
    res_w = image.width - width
    res_h = image.height - height
    l_crop, top_crop = res_w//2, res_h//2
    return image.crop((l_crop, top_crop, width + l_crop, top_crop + height))


class CenterCrop:
    def __init__(self, height, width):
        """
        Class used for Pytorch center cropping.
        :param height: Image crop height.
        :param width: Image crop width.
        :param scale: Scale factor to enlarge from
        """
        self.height = height
        self.width = width

    def __call__(self, image):
        """
        Args:
            image (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return center_crop(image, self.height, self.width)


def random_matched_crop(image_lr, image_hr, crop_size, scale):
    rnd_h = random.randint(0, max(0, image_lr.size()[1] - crop_size))
    rnd_w = random.randint(0, max(0, image_lr.size()[2] - crop_size))
    cropped_lr = image_lr[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size]
    rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
    cropped_hr = image_hr[:, rnd_h_GT:rnd_h_GT + int(crop_size*scale), rnd_w_GT:rnd_w_GT + int(crop_size*scale)]
    return cropped_lr, cropped_hr


def random_flip_rotate(*img, hflip=True, rot=True):
    # Modified from https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/data/util.py
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = torch.flip(img, [2])
        if vflip:
            img = torch.flip(img, [1])
        if rot90:
            img = torch.transpose(img, 1, 2)
        return img

    return [_augment(I) for I in img]



