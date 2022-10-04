import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def conv_drop_block(in_channels, out_channels, drop_p):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(drop_p),
        nn.MaxPool2d(2)
    )


class Conv4Proto(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, drop_p=0.):
        super(Conv4Proto, self).__init__()
        self.drop_p = drop_p

        kwargs = {}
        if self.drop_p > 0.:
            conv = conv_drop_block
            kwargs['drop_p'] = self.drop_p
        else:
            conv = conv_block

        self.encoder = nn.Sequential(
            conv(x_dim, hid_dim, **kwargs),
            conv(hid_dim, hid_dim, **kwargs),
            conv(hid_dim, hid_dim, **kwargs),
            conv(hid_dim, z_dim, **kwargs),
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)
