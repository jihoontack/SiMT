from collections import OrderedDict

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(mode, P):
    if P.regression:
        fname = f'{P.dataset}_{P.model}_{mode}_{P.num_shots}shot_{P.batch_size}task'
    else:
        fname = f'{P.dataset}_{P.model}_{mode}_{P.num_ways}way_{P.num_shots}shot_{P.batch_size}task'

    if mode in ['maml', 'metasgd', 'anil']:
        if P.simt:
            from train.gradient_based.maml_simt import maml_step as train_func
            from train.gradient_based.maml_simt import check
            fname += f'_eta{P.eta}_lam{P.lam}_{P.drop_p}drop'
            if not P.regression:
                fname += f'_{P.temp}temp'
        else:
            from train.gradient_based.maml import maml_step as train_func
            from train.gradient_based.maml import check

    elif mode == 'protonet':
        assert not P.regression
        if P.simt:
            from train.metric_based.protonet_simt import protonet_step as train_func
            from train.metric_based.protonet_simt import check
            fname += f'_eta{P.eta}_lam{P.lam}_{P.drop_p}drop_{P.temp}temp'
        else:
            from train.metric_based.protonet import protonet_step as train_func
            from train.metric_based.protonet import check

    else:
        raise NotImplementedError()

    today = check(P)
    if P.baseline:
        today = False

    fname += f'_seed_{P.seed}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train_func, fname, today


def copy_model_param(model, params=None):
    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    copy_params = OrderedDict()

    for (name, param) in params.items():
        copy_params[name] = param.clone().detach()
        copy_params[name].requires_grad_()

    return copy_params


def param_ema(P, model):
    params = OrderedDict(model.meta_named_parameters())

    for (name, param) in params.items():
        P.moving_average[name].data = P.eta * P.moving_average[name].data + (1 - P.eta) * param.data

    if 'metasgd' in P.mode:
        for (name, param) in P.inner_lr.items():
            P.moving_inner_lr[name].data = P.eta * P.moving_inner_lr[name].data + (1 - P.eta) * param.data


def dropout_eval(m):
    def _is_leaf(model):
        return len(list(model.children())) == 0
    if hasattr(m, 'dropout'):
        m.dropout.eval()

    for child in m.children():
        if not _is_leaf(child):
            dropout_eval(child)


def function_consistency(P, outputs, outputs_target, temp):
    if P.regression:
        return F.mse_loss(outputs, outputs_target)
    p_s = F.log_softmax(outputs / temp, dim=1)
    p_t = F.softmax(outputs_target / temp, dim=1)
    return F.kl_div(p_s, p_t, size_average=False) * (temp ** 2) / outputs.shape[0]
