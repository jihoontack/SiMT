from collections import OrderedDict

import torch
from torchmeta.modules import MetaModule


def maml_inner_adapt(model, criterion, inputs, targets, step_size, num_steps,
                     first_order=False, params=None):

    """ inner gradient step """
    for step_inner in range(num_steps):
        outputs_train = model(inputs, params=params)
        loss = criterion(outputs_train, targets)

        model.zero_grad()
        params = gradient_update_parameters(
            model, loss, params=params, step_size=step_size, first_order=first_order
        )

    return params, loss


def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))
    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order,
                                allow_unused=True)  # this is for anil implementation

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = 0.
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = 0.
            updated_params[name] = param - step_size * grad

    return updated_params
