from models.metamodel.conv4 import MetaConvModel
from models.metamodel.resnet12 import ResNet12
from models.metamodel.conv_regress import ConvRegressor
from models.protonet_model.conv4 import Conv4Proto
from models.protonet_model.resnet12 import ResNetProto12


def ModelConvMiniImagenet(out_features, hidden_size=64, **kwargs):
    return MetaConvModel(3, out_features, hidden_size=hidden_size, feature_size=5 * 5 * hidden_size, **kwargs)


def get_model(P, modelstr):
    kwargs = {'drop_p': P.drop_p}
    if modelstr == 'conv4':
        if 'protonet' in P.mode:
            model = Conv4Proto(**kwargs)
        else:
            model = ModelConvMiniImagenet(P.num_ways, **kwargs)
    elif modelstr == 'resnet12':
        if 'protonet' in P.mode:
            model = ResNetProto12(**kwargs)
        else:
            model = ResNet12(P.num_ways, **kwargs)
    elif modelstr == 'conv_pose':
        model = ConvRegressor(dataset=P.dataset, **kwargs)
    else:
        raise NotImplementedError()

    if P.mode == 'anil':
        model.anil = True

    return model
