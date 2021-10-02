from .architectures import *
from SISR.models.attention_manipulators import QModel
import numpy as np
import time


class QRCANHandler(QModel):
    """
    Meta-modified QRCAN.  Standard setup consists of 10 residual groups, each with 20 residual blocks.

    Meta-attention can be selectively inserted in different layers using the include_q_layer,
    selective_meta_blocks and num_q_layers_inner_residual parameters:
    include_q_layer: Bool; set to True to insert q-layers within network residual blocks.
    selective_meta_blocks: List of Bools; must be the same length as the number of Residual Groups in RCAN.
    Setting an element of the list to True will signal the addition of meta-layers in the corresponding residual block.
    q_layers_inner_residual: Number of q_layers to add within each residual block.
    Set to None to add q_layers to all inner residual blocks.
    Otherwise, other parameters controlling the network can be set akin to normal RCAN.

    Check QRCAN architecture class for further info on internal parameters available.
    """
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, scheduler=None,
                 scheduler_params=None, style='modulate', perceptual=None, clamp=False, min_mu=-0.2,
                 max_mu=0.8, n_feats=64, **kwargs):
        super(QRCANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)

        self.net = QRCAN(scale=scale, in_feats=in_features, num_metadata=self.num_metadata,
                         n_feats=n_feats, style=style, **kwargs)
        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.model_name = 'qrcan'
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.base_scaler = np.linspace(0, 1, n_feats)
        self.clamp = clamp
        self.style = style

    @staticmethod
    def gaussian(x, mu, sig=0.2):
        return torch.from_numpy((1/(np.sqrt(2*np.pi)*sig)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))).type(torch.float32)

    def scale_qpi(self, qpi):  # TODO: various updates possible here
        scaled_qpi = (qpi*(self.max_mu-self.min_mu)) + self.min_mu
        scalers = []
        for i in range(scaled_qpi.size(0)):
            scalers.append(self.gaussian(self.base_scaler, scaled_qpi[i].squeeze().numpy()))
        full_scalers = torch.stack(scalers)
        if self.clamp:
            full_scalers = torch.clamp(full_scalers, 0, 1)
        return full_scalers.unsqueeze(2).unsqueeze(3)


class QEDSRHandler(QModel):
    """
    Meta-modified EDSR.  Check original EDSR handler/architecture for details on inputs.
    """
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, num_blocks=16,
                 num_features=64, res_scale=0.1, scheduler=None, scheduler_params=None,
                 perceptual=None, **kwargs):
        super(QEDSRHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)

        self.net = QEDSR(scale=scale, in_features=in_features, num_features=num_features, num_blocks=num_blocks,
                         res_scale=res_scale, input_para=self.num_metadata, **kwargs)

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.model_name = 'qedsr'
        self.criterion = nn.L1Loss()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)


class QSANHandler(QModel):
    """
    Meta-modified SAN.  Check original SAN handler/architecture for details on inputs.
    """
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 max_combined_im_size=160000, scheduler=None, scheduler_params=None, **kwargs):
        super(QSANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)

        self.net = QSAN(scale=scale, input_para=self.num_metadata)
        self.scale = scale
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.max_combined_im_size = max_combined_im_size

        self.model_name = 'qsan'

    def forward_chop(self, x, extra_channels, shave=10):
        # modified from https://github.com/daitao/SAN/blob/master/TestCode/code/model/__init__.py

        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < self.max_combined_im_size:
            sr_list = []
            for chunk in lr_list:
                sr_list.append(self.run_chopped_eval(chunk, extra_channels))
        else:
            sr_list = [
                self.forward_chop(patch, extra_channels, shave=shave)
                for patch in lr_list]

        h, w = self.scale * h, self.scale * w
        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size
        shave *= self.scale

        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None, timing=False, *args, **kwargs):
        extra_channels = self.generate_channels(x, metadata, metadata_keys).to(self.device)
        if timing:
            tic = time.perf_counter()
        sr_image = self.forward_chop(x, extra_channels)
        if timing:
            toc = time.perf_counter()

        if request_loss:
            return sr_image, self.criterion(sr_image, y), toc - tic if timing else None
        else:
            return sr_image, None, toc - tic if timing else None

    def run_chopped_eval(self, x, extra_channels):
        return super().run_eval(x, y=None, request_loss=False, extra_channels=extra_channels)[0]


class QHANHandler(QModel):
    """
    Meta-modified HAN.  Check original HAN handler/architecture for details on inputs.
    """
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 scheduler=None, scheduler_params=None, **kwargs):
        super(QHANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)

        self.net = QHAN(scale=scale, num_metadata=self.num_metadata)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.model_name = 'qhan'
