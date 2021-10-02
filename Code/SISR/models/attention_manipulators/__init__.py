from SISR.models import BaseModel
import torch
import numpy as np


class QModel(BaseModel):
    """
    Model template for use when modulating typical SR networks with metadata.
    """
    def __init__(self, metadata=None, **kwargs):
        self.style = None  # only relevant to QRCAN
        self.channel_concat = False  # only relevant to models concatenating extra channels with input
        if metadata is not None:
            self.num_metadata = len(metadata)  # TODO: better way to remove this hardcoding? - NEEDS FIXING, DOES NOT WORK IN ALL SCENARIOS
            if 'all' in metadata:
                self.num_metadata += 39  # all celeba attributes

            if 'blur_kernel' in metadata:
                self.num_metadata += 9
            elif 'unmodified_blur_kernel' in metadata:
                self.num_metadata += 440

            self.metadata = metadata
        else:
            self.metadata = ['qpi']
            self.num_metadata = 1

        super(QModel, self).__init__(**kwargs)

    def generate_channels(self, x, metadata, keys):
        """
        Specific function used to morph metadata into format required for Q-layer blocks.
        """
        if metadata is None:
            raise RuntimeError('Metadata needs to be specified for this network to run properly.')
        extra_channels = torch.ones(x.size(0), self.num_metadata)
        if 'all' in self.metadata:
            mask = [True] * self.num_metadata
        else:
            mask = [True if key[0] in self.metadata else False for key in keys]

        for index, _ in enumerate(extra_channels):
            if len(keys) == 1:  # TODO: any way to shorten this?
                added_info = metadata[index]
            else:
                added_info = metadata[index][mask]
            extra_channels[index, ...] = extra_channels[index, :] * added_info
        extra_channels = extra_channels.unsqueeze(2).unsqueeze(3)
        if self.style == 'modulate':
            extra_channels = self.scale_qpi(extra_channels)
        return extra_channels

    def generate_sft_channels(self, x, metadata, metadata_keys):
        """
        Specific function used to morph metadata into format required for SFT blocks.
        """
        if metadata is None:
            raise RuntimeError('Metadata needs to be specified for this network to run properly.')
        extra_channels = torch.ones(x.size(0), self.num_metadata, *x.size()[2:])
        if extra_channels.device != metadata.device:
            extra_channels = extra_channels.to(metadata.device)
        mask = [True if key[0] in self.metadata else False for key in metadata_keys]
        for index, _ in enumerate(extra_channels):

            if len(metadata_keys) == 1:  # TODO: any way to shorten this?
                added_info = metadata[index]
            else:
                added_info = metadata[index][mask]
            if self.num_metadata == 1:
                extra_channels[index, ...] = extra_channels[index, ...] * added_info
            else:
                if isinstance(metadata, np.ndarray):
                    extra_channels[index, ...] = torch.from_numpy(np.expand_dims(
                        np.expand_dims(added_info, axis=-1), axis=-1).repeat(x.size()[2], 1).repeat(
                        x.size()[3], 2))
                else:
                    extra_channels[index, ...] = torch.unsqueeze(torch.unsqueeze(added_info, -1), -1).repeat_interleave(
                        x.size()[2], 1).repeat_interleave(x.size()[3], 2)

        return extra_channels

    def channel_concat_logic(self, x, extra_channels, metadata, metadata_keys):
        """
        Main channel concatenation stage.
        Metadata needs to be selectively filtered and converted to the correct format for the model to use.
        :param x: Input image batch (N, C, H, W)
        :param extra_channels: Optional pre-prepared metadata channels.  Can be set to None to ignore.
        :param metadata: Metadata information, for each image in batch provided (N, M).
        :param metadata_keys: List of keys corresponding to metadata, to allow for selective filtering.
        :return: Modulated input batch (if required), modulated metadata ready for model use.
        """
        if extra_channels is None:
            extra_channels = self.generate_channels(x, metadata, metadata_keys)
            if not self.channel_concat and self.device != extra_channels.device:
                extra_channels = extra_channels.to(self.device)

        if self.channel_concat:
            input_data = torch.cat((x, extra_channels), 1)
        else:
            input_data = x

        return input_data, extra_channels

    def run_train(self, x, y, metadata=None, extra_channels=None, metadata_keys=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        return super().run_train(input_data, y, extra_channels=extra_channels, **kwargs)

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None,
                 extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        return super().run_eval(input_data, y, request_loss=request_loss, extra_channels=extra_channels, **kwargs)

    def run_forensic(self, x, metadata=None, metadata_keys=None, extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        return super().run_forensic(input_data, qpi=extra_channels)

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        return self.net.forward(x, metadata=extra_channels)
