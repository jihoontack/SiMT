import time

import torch
import torch.nn as nn

from torchmeta.utils.prototype import get_prototypes

from train import function_consistency
from train.metric_based import get_accuracy
from models.model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dropout_eval(model):
    for m in model.modules():
        if type(m) == nn.Dropout:
            m.eval()


def dropout_train_res(model, train_mode=True):
    for m in model.modules():
        if hasattr(m, 'dropout_train'):
            m.dropout_train = train_mode


def param_ema(P, model):
    for param_q, param_k in zip(model.parameters(), P.moving_average.parameters()):
        param_k.data = param_k.data * P.eta + param_q.data * (1. - P.eta)


def check(P):
    filename_with_today_date = True
    assert P.num_shots_global == 0
    return filename_with_today_date


def protonet_step(P, step, model, criterion, optimizer, batch, metric_logger, logger):

    stime = time.time()

    assert not P.regression
    if not hasattr(P, 'moving_average'):
        P.moving_average = get_model(P, P.model).to(device)
        for param_q, param_k in zip(model.parameters(), P.moving_average.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    model.train()
    P.moving_average.train()

    train_inputs, train_targets = batch['train']
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_embeddings = model(train_inputs)
    P.moving_average.train()  # use dropout
    if P.model == 'resnet12_protonet':
        dropout_train_res(P.moving_average)
    with torch.no_grad():
        train_embeddings_ema = P.moving_average(train_inputs)

    test_inputs, test_targets = batch['test']
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    test_embeddings = model(test_inputs)
    dropout_eval(P.moving_average)  # do not use dropout, but utilize bn train mode
    if P.model == 'resnet12_protonet':
        dropout_train_res(P.moving_average, False)
    with torch.no_grad():
        test_embeddings_ema = P.moving_average(test_inputs).detach()

    prototypes = get_prototypes(train_embeddings, train_targets, P.train_num_ways)
    squared_distances = - torch.sum((prototypes.unsqueeze(2)
                                    - test_embeddings.unsqueeze(1)) ** 2, dim=-1)

    prototypes_ema = get_prototypes(train_embeddings_ema, train_targets, P.train_num_ways)
    squared_distances_ema = - torch.sum((prototypes_ema.unsqueeze(2)
                                         - test_embeddings_ema.unsqueeze(1)) ** 2, dim=-1)

    loss_query = (1. - P.lam) * criterion(squared_distances, test_targets)
    loss_teacher = P.lam * function_consistency(P, squared_distances, squared_distances_ema, P.temp)

    loss = loss_query + loss_teacher

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    """ exponential weight average """
    with torch.no_grad():
        param_ema(P, model)

    acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

    """ track stat """
    metric_logger.meters['batch_time'].update(time.time() - stime)
    metric_logger.meters['meta_test_cls'].update(loss.item())
    metric_logger.meters['meta_reg_loss'].update(loss_teacher.item())
    metric_logger.meters['train_acc'].update(acc)

    if step % P.print_step == 0:
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary('train/meta_test_cls',
                              metric_logger.meta_test_cls.global_avg, step)
        logger.scalar_summary('train/train_acc',
                              metric_logger.train_acc.global_avg, step)
        logger.scalar_summary('train/meta_reg_loss',
                              metric_logger.meta_reg_loss.global_avg, step)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, step)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[MetaTestLoss %f] [MetaRegLoss %f]' %
                   (step, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.meta_test_cls.global_avg, metric_logger.meta_reg_loss.global_avg))
