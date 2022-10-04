import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d,
                               MetaSequential, MetaLinear)


def conv_no_drop_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('relu', nn.ReLU()),
    ]))


def conv_drop_block(in_channels, out_channels, drop, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(drop)),
    ]))


def conv_feature_no_drop_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1,
                                track_running_stats=False)),
        ('relu', nn.ReLU()),
    ]))


def conv_feature_drop_block(in_channels, out_channels, drop, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1,
                                track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(drop)),
    ]))


class ConvRegressor(MetaModule):
    """
        MAML Model used for Pascal1D and ShapeNet1D.
    """
    def __init__(self, drop_p=0., dataset='pose'):
        super(ConvRegressor, self).__init__()
        self.img_size = [128, 128, 1]
        self.img_channels = self.img_size[2]
        self.dim_hidden = 64
        self.dim_w = 196
        self.img_w_size = int(np.sqrt(self.dim_w))
        self.anil = False
        self.drop_p = drop_p
        if dataset == 'pose':
            self.output_dim = 1
        elif dataset == 'shapenet':
            self.output_dim = 2
        else:
            raise NotImplementedError()

        kwargs = {}
        if self.drop_p > 0.:
            kwargs['drop'] = self.drop_p
            conv_block = conv_drop_block
            conv_feature_block = conv_feature_drop_block
        else:
            conv_block = conv_no_drop_block
            conv_feature_block = conv_feature_no_drop_block

        self.encoder_w = MetaSequential(OrderedDict([
            ('layer1', conv_block(self.img_channels, 32, kernel_size=3,
                                  stride=2, padding=1, bias=True, **kwargs)),
            ('layer2', conv_block(32, 48, kernel_size=3,
                                  stride=2, padding=1, bias=True, **kwargs)),
            ('pool', nn.MaxPool2d((2, 2))),
            ('layer3', conv_block(48, 64, kernel_size=3,
                                  stride=2, padding=1, bias=True, **kwargs)),
            ('flatten', nn.Flatten()),
            ('linear', MetaLinear(4096, self.dim_w, bias=True))
        ]))

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_feature_block(self.img_channels, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True, **kwargs)),
            ('layer2', conv_feature_block(self.dim_hidden, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True, **kwargs)),
            ('layer3', conv_feature_block(self.dim_hidden, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True, **kwargs)),
            ('layer4', conv_feature_block(self.dim_hidden, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True, **kwargs)),
            ('pool', nn.AdaptiveAvgPool2d(1)),
        ]))

        if dataset == 'shapenet':
            self.regressor = MetaSequential(OrderedDict([
                ('regressor', MetaLinear(self.dim_hidden, self.output_dim, bias=True)),
                ('tanh', nn.Tanh()),
            ]))
        else:
            self.regressor = MetaLinear(self.dim_hidden, self.output_dim, bias=True)

    def forward(self, inputs, params=None):
        if self.anil:
            params_encoder_w = None
            params_features = None
        else:
            params_encoder_w = self.get_subdict(params, 'encoder_w')
            params_features = self.get_subdict(params, 'features')

        bs = inputs.shape[0]
        inputs = inputs.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
        inputs = self.encoder_w(inputs, params=params_encoder_w).reshape(-1, self.img_channels,
                                                                         self.img_w_size,
                                                                         self.img_w_size)
        inputs = self.features(inputs, params=params_features).reshape(bs, self.dim_hidden)
        outputs = self.regressor(inputs, params=self.get_subdict(params, 'regressor'))

        return outputs
