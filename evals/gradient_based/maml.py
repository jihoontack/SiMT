import math

import torch

from train.gradient_based import maml_inner_adapt
from data.shapenet1d import degree_loss
from evals import accuracy
from utils import MetricLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    filename_with_today_date = True
    return filename_with_today_date


def test_classifier(P, model, loader, criterion, steps, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()
    acc = 0.

    for n, batch in enumerate(loader):

        if n * P.test_batch_size > P.max_test_task:
            break

        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.to(device, non_blocking=True)
        train_targets = train_targets.to(device, non_blocking=True)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device, non_blocking=True)
        test_targets = test_targets.to(device, non_blocking=True)

        for task_idx, (train_input, train_target, test_input, test_target) \
                in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):

            params, loss_train = maml_inner_adapt(
                model, criterion, train_input, train_target, P.inner_lr, P.inner_steps_test, first_order=True
            )

            """ outer loss aggregate """
            with torch.no_grad():
                outputs_test = model(test_input, params=params)
            loss = criterion(outputs_test, test_target)

            if not P.regression:
                acc = accuracy(outputs_test, test_target, topk=(1, ))[0].item()
            elif P.dataset == 'shapenet':
                acc = - degree_loss(outputs_test, test_target).item()
            elif P.dataset == 'pose':
                acc = - loss.item()
            else:
                raise NotImplementedError()

            metric_logger.meters['loss_train'].update(loss_train.item())
            metric_logger.meters['loss'].update(loss.item())
            metric_logger.meters['acc'].update(acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log_(' * [Acc@1 %.3f] [LossOut %.3f] [LossIn %.3f]' %
         (metric_logger.acc.global_avg, metric_logger.loss.global_avg,
          metric_logger.loss_train.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/acc', metric_logger.acc.global_avg, steps)
        logger.scalar_summary('eval/loss_test', metric_logger.loss.global_avg, steps)
        logger.scalar_summary('eval/loss_train', metric_logger.loss_train.global_avg, steps)

    model.train(mode)

    return metric_logger.acc.global_avg
