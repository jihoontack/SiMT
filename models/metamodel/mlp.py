import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)


class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].
    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).
    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes, drop_p=0.):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.drop_p = drop_p

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(
            OrderedDict([('layer{0}'.format(i + 1),
                          MetaSequential(OrderedDict([
                              ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                              ('relu', nn.ReLU())
                          ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])])
        )
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)
        self.anil = False
        if self.drop_p > 0.:
            self.dropout = nn.Dropout(self.drop_p)

    def forward(self, inputs, params=None):
        if self.anil:
            params_feature = None
        else:
            params_feature = self.get_subdict(params, 'features')

        features = self.features(inputs, params=params_feature)
        if self.drop_p > 0.:
            features = self.dropout(features)

        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits
