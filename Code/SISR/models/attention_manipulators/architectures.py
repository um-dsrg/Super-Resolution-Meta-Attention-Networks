from collections import OrderedDict
from torch import nn
import torch
import numpy as np

from SISR.models.attention_manipulators.q_layer import ParaCALayer
from SISR.models.advanced.architectures import common
from SISR.models.advanced.SAN_blocks import Nonlocal_CA
from SISR.models.attention_manipulators.qsan_blocks import QLSRAG
from SISR.models.advanced.HAN_blocks import LAM_Module, CSAM_Module


class PALayer(nn.Module):
    # adapted from https://github.com/zhilin007/FFA-Net/blob/master/net/models/FFA.py
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

    def forensic(self, x):
        y = self.pa(x)
        return x*y, y.cpu().data.numpy().squeeze()


# Channel Attention (CA) Layer
class QCALayer(nn.Module):
    """
    Combined channel-attention and meta-attention layer.  Diverse style choices available.
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, channel, style, reduction=16, num_metadata=1):
        """
        :param channel:  Network feature map channel count.
        :param style: Type of attention to use.  Options are:
        modulate:  Normal channel attention occurs, but meta-vector is multiplied with the final attention
        vector prior to network modulation.
        mini_concat:  Concatenate meta-vector with inner channel attention vector.
        max_concat:  Concatenate meta-vector with feature map aggregate, straight after average pooling.
        softmax:  Implements max_concat, but also applies softmax after the final FC layer.
        extended_attention: Splits attention into four layers, and adds metadata vector in second layer.
        standard:  Do not introduce any metadata.
        :param reduction: Level of downscaling to use for inner channel attention vector.
        :param num_metadata: Expected metadata input size.
        """
        super(QCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight

        if reduction < 16:
            raise RuntimeError('Using an extreme channel attention reduction value')

        if style == 'modulate' or style == 'mini_concat' or style == 'standard':
            channel_in = channel
        else:
            channel_in = channel + num_metadata

        channel_reduction = channel // reduction

        if style == 'modulate' or style == 'max_concat' or style == 'softmax' or style == 'standard':
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel_in, channel_reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif style == 'mini_concat':
            self.pre_concat = nn.Conv2d(channel_in, channel_reduction, 1, padding=0, bias=True)
            self.conv_du = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_reduction + num_metadata, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif style == 'extended_attention':

            channel_fractions = [(channel_in, channel//2),
                                 (channel//2 + num_metadata, channel//4),
                                 (channel//4 + num_metadata, channel_reduction)]
            self.feature_convs = nn.ModuleList()
            for (inp, outp) in channel_fractions:
                self.feature_convs.append(
                    nn.Sequential(
                        nn.Conv2d(inp, outp, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True)
                    )
                )
            self.final_conv = nn.Sequential(
                nn.Conv2d(channel_reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        if style == 'softmax':
            self.softmax = nn.Softmax(dim=1)

        self.style = style

    def forward(self, x, attributes):

        y = self.avg_pool(x)
        if self.style == 'modulate':
            y = self.conv_du(y) * attributes
        elif self.style == 'max_concat':
            y = self.conv_du(torch.cat((y, attributes), dim=1))
        elif self.style == 'mini_concat':
            y = self.pre_concat(y)
            y = self.conv_du(torch.cat((y, attributes), dim=1))
        elif self.style == 'extended_attention':
            for conv_section in self.feature_convs:
                y = conv_section(torch.cat((y, attributes), dim=1))
            y = self.final_conv(y)
        elif self.style == 'softmax':
            y = self.conv_du(torch.cat((y, attributes), dim=1))
            y = self.softmax(y)
        elif self.style == 'standard':
            y = self.conv_du(y)
        else:
            raise NotImplementedError

        return x * y

    def forensic(self, x, attributes):
        inner_forensic_data = {}
        y = self.avg_pool(x)
        if self.style == 'standard':
            inner_forensic_data['inner_vector'] = self.conv_du[1](self.conv_du[0](y)).cpu().data.numpy().squeeze()
            y = self.conv_du(y)
        else:
            inner_forensic_data['inner_vector'] = self.conv_du[1](self.conv_du[0](torch.cat((y, attributes), dim=1))).cpu().data.numpy().squeeze()
            y = self.conv_du(torch.cat((y, attributes), dim=1))

        inner_forensic_data['mask_multiplier'] = y.cpu().data.numpy().squeeze()

        return x * y, inner_forensic_data


# Residual Channel Attention Block (RCAB)
class QRCAB(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(
            self, conv, n_feat, kernel_size, reduction, style='modulate', pa=False, q_layer=False,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, num_metadata=1):

        super(QRCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        self.final_body = QCALayer(channel=n_feat, reduction=reduction, style=style, num_metadata=num_metadata)
        self.pa = pa
        self.q_layer = q_layer
        if pa:
            self.pa_node = PALayer(channel=n_feat)
        if q_layer:
            self.q_node = ParaCALayer(network_channels=n_feat, num_metadata=num_metadata, nonlinearity=True)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x[0])
        res = self.final_body(res, x[1])
        if self.pa:
            res = self.pa_node(res)
        if self.q_layer:
            res = self.q_node(res, x[1])
        res += x[0]
        return res, x[1]

    def forensic(self, x, qpi):

        res = self.body(x)
        conv_data = []
        for module in self.body:
            if isinstance(module, nn.Conv2d):
                conv_data.append(module.weight.detach().cpu().numpy().flatten())

        res, forensic_data = self.final_body.forensic(res, qpi)
        if self.pa:
            res, forensic_pa = self.pa_node.forensic(res)
            forensic_data['pixel_attention_map'] = forensic_pa
        if self.q_layer:
            res, meta_attention = self.q_node.forensic(res, qpi)
            forensic_data['meta_attention_map'] = meta_attention

        forensic_data['pre-residual'] = res
        forensic_data['pre-residual-flat'] = res.cpu().numpy().flatten()
        res += x
        forensic_data['post-residual'] = res
        forensic_data['post-residual-flat'] = res.cpu().numpy().flatten()
        forensic_data['conv_flat'] = np.hstack(np.array(conv_data))
        return res, forensic_data


# Residual Group (RG)
class QResidualGroup(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, style, num_metadata,
                 pa, q_layer, num_q_layers):
        super(QResidualGroup, self).__init__()
        modules_body = []

        for index in range(n_resblocks):
            if num_q_layers is None or index < num_q_layers:
                q_in = q_layer
            else:
                q_in = False
            modules_body.append(QRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                                      act=act, res_scale=res_scale, style=style,
                                      pa=pa, q_layer=q_in, num_metadata=num_metadata))

        self.final_body = conv(n_feat, n_feat, kernel_size)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res, _ = self.body(x)
        res = self.final_body(res)
        res += x[0]
        return res, x[1]

    def forensic(self, x, qpi):
        res = x
        forensic_data = []
        for module in self.body:
            res, RCAB_data = module.forensic(res, qpi)
            forensic_data.append(RCAB_data)
        res = self.final_body(res)
        res += x
        return res, forensic_data


class QRCAN(nn.Module):
    def __init__(self, n_resblocks=20, n_resgroups=10, n_feats=64, in_feats=3, out_feats=3, scale=4, reduction=16,
                 res_scale=1.0, style='modulate', num_metadata=1, include_pixel_attention=False,
                 selective_meta_blocks=None, num_q_layers_inner_residual=None, include_q_layer=False, **kwargs):
        """
        Main QRCAN architecture - make sure to check handler docstring for more info on this model.
        :param n_resblocks: Number of residual blocks (within each residual group).
        :param n_resgroups: Number of residual groups (following original paper nomenclature).
        :param n_feats: Number of channels within network conv layers.
        :param in_feats: Input features (if RGB, leave at 3 channels).
        :param out_feats: Output features (if RGB, leave at 3 channels).
        :param scale: SR scale.
        :param reduction: Magnitude of reduction for channel attention (e.g. for a reduction of 16 and a n_feats size of 64,
         internal channel vector will have a size of 4.
        :param res_scale: Scale factor to apply to each residual group.  By default not used (set to 1).
        :param style: Channel attention style, if combining channel attention with meta-attention.
        :param num_metadata: If using meta-attention, indicate the expected metadata vector size here.
        :param include_pixel_attention: Set to true to include pixel attention after each residual block.
        :param selective_meta_blocks:  *Check QRCAN handler for more info*
        :param num_q_layers_inner_residual: *Check QRCAN handler for more info*
        :param include_q_layer: *Check QRCAN handler for more info*
        """
        super(QRCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        self.style = style

        # define head module
        modules_head = [common.default_conv(in_feats, n_feats, kernel_size)]

        # define body module
        if selective_meta_blocks is None:
            modules_body = [
                QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                               num_metadata=num_metadata, pa=include_pixel_attention, q_layer=include_q_layer,
                               act=act, res_scale=res_scale, n_resblocks=n_resblocks,
                               num_q_layers=num_q_layers_inner_residual) for _ in range(n_resgroups)]
        else:
            modules_body = []

            for index in range(n_resgroups):
                if selective_meta_blocks[index]:
                    include_q = include_q_layer
                else:
                    include_q = False
                modules_body.append(
                    QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                                   num_metadata=num_metadata, pa=include_pixel_attention, q_layer=include_q,
                                   act=act, res_scale=res_scale, n_resblocks=n_resblocks,
                                   num_q_layers=num_q_layers_inner_residual))

        self.final_body = common.default_conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, out_feats, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, metadata):
        x = self.head(x)
        res, *_ = self.body((x, metadata))
        res = self.final_body(res)
        res += x
        x = self.tail(res)

        return x

    def forensic(self, x, qpi, *args, **kwargs):
        x = self.head(x)
        data = OrderedDict()
        res = x
        for index, module in enumerate(self.body):
            res, res_forensic_data = module.forensic(res, qpi)
            for rcab_index, rcab_forensic_data in enumerate(res_forensic_data):
                data['R%d.C%d' % (index, rcab_index)] = rcab_forensic_data
        res = self.final_body(res)
        res += x
        x = self.tail(res)
        return x, data


class ParamResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, n_params, kernel_size, act=nn.ReLU(True),
            bias=True, res_scale=1.0, q_layer_nonlinearity=False):

        super(ParamResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.attention_layer = ParaCALayer(n_feats, n_params, nonlinearity=q_layer_nonlinearity)

        self.res_scale = res_scale

    def forward(self, x):

        params = x[1]
        res = self.body(x[0])
        res = res.mul(self.res_scale)
        res = self.attention_layer(res, x[1])
        res += x[0]

        return res, params


class QEDSR(nn.Module):
    """
    modified EDSR to allow insertion of meta-attention.  Refer to original EDSR for info on function inputs.
    """
    def __init__(self,
                 in_features=3, out_features=3, num_features=64, input_para=1,
                 num_blocks=16, scale=4, res_scale=0.1, q_layer_nonlinearity=False, **kwargs):
        super(QEDSR, self).__init__()

        n_resblocks = num_blocks
        n_feats = num_features
        kernel_size = 3

        # define head module
        self.head = common.default_conv(in_features, n_feats, kernel_size)

        # define body module
        m_body = [
            ParamResBlock(
                common.default_conv, n_feats, input_para, kernel_size, res_scale=res_scale,
                q_layer_nonlinearity=q_layer_nonlinearity
            ) for _ in range(n_resblocks)
        ]
        self.final_body = common.default_conv(n_feats, n_feats, kernel_size)

        # define tail module
        m_tail = [
            common.Upsampler(common.default_conv, scale, n_feats),
            common.default_conv(n_feats, out_features, kernel_size)
        ]

        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, metadata):
        x = self.head(x)
        res, _ = self.body((x, metadata))
        res = self.final_body(res)
        res += x
        x = self.tail(res)
        return x


class QSAN(nn.Module):
    """
    modified QSAN to allow insertion of meta-attention.  Refer to original SAN model for info on parameters.
    """
    def __init__(self, n_resgroups=20, n_resblocks=10, n_feats=64, reduction=16, scale=4, rgb_range=255, n_colors=3,
                 res_scale=1, conv=common.default_conv, input_para=1, **kwargs):
        super(QSAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale
        act = nn.ReLU(inplace=True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups
        self.RG = nn.ModuleList([QLSRAG(conv, n_feats, kernel_size, reduction, num_metadata=input_para,
                                       act=act, res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)])
        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8, reduction=8, sub_sample=False, bn_layer=False)

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)

    def forward(self, x, metadata):

        x = self.head(x)

        # add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # share-source residual gruop
        for i,l in enumerate(self.RG):
            xx = l((xx, metadata))[0] + self.gamma * residual

        # add nonlocal
        res = self.non_local(xx)
        res = res + x

        x = self.tail(res)

        return x


class QHAN(nn.Module):
    """
    Modified HAN network to include meta-attention.  Refer to original model for info on parameters.
    """
    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16, num_metadata=0,
                 scale=4, n_colors=3, res_scale=1.0, conv=common.default_conv,  num_q_layers_inner_residual=None):
        super(QHAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            QResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, style='standard',
                num_metadata=num_metadata, pa=False, q_layer=True, n_resblocks=n_resblocks,
                num_q_layers=num_q_layers_inner_residual)
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, metadata):

        x = self.head(x)
        res = x

        for name, midlayer in self.body._modules.items():
            if type(midlayer).__name__ == 'QResidualGroup':
                res, _ = midlayer((res, metadata))
            else:
                res = midlayer(res)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)

        out1 = res

        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)

        res += x

        x = self.tail(res)

        return x
