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
    acc_ema = 0.

    if hasattr(P, 'moving_inner_lr'):
        inner_step_ema = P.moving_inner_lr
    else:
        inner_step_ema = P.inner_lr

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
                model, criterion, train_input, train_target, P.inner_lr, P.inner_steps_test,
                first_order=True
            )

            params_ema, loss_train_ema = maml_inner_adapt(
                model, criterion, train_input, train_target, inner_step_ema, P.inner_steps_test,
                first_order=True, params=P.moving_average
            )

            """ outer loss aggregate """
            with torch.no_grad():
                outputs_test = model(test_input, params=params)
                outputs_test_ema = model(test_input, params=params_ema)
            loss = criterion(outputs_test, test_target)
            loss_ema = criterion(outputs_test_ema, test_target)

            if not P.regression:
                acc = accuracy(outputs_test, test_target, topk=(1, ))[0].item()
                acc_ema = accuracy(outputs_test_ema, test_target, topk=(1, ))[0].item()
            elif P.dataset == 'shapenet':
                acc = - degree_loss(outputs_test, test_target).item()
                acc_ema = - degree_loss(outputs_test_ema, test_target).item()
            elif P.dataset == 'pose':
                acc = - loss.item()
                acc_ema = - loss_ema.item()
            else:
                raise NotImplementedError()

            metric_logger.meters['loss_train_ori'].update(loss_train.item())
            metric_logger.meters['loss_ori'].update(loss.item())
            metric_logger.meters['acc_ori'].update(acc)

            metric_logger.meters['loss_train'].update(loss_train_ema.item())
            metric_logger.meters['loss'].update(loss_ema.item())
            metric_logger.meters['acc'].update(acc_ema)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log_(' * [AccEMA@1 %.3f] [LossOutEMA %.3f] [LossInEMA %.3f]' %
         (metric_logger.acc.global_avg, metric_logger.loss.global_avg,
          metric_logger.loss_train.global_avg))
    log_(' * [Acc@1 %.3f] [LossOut %.3f] [LossIn %.3f]' %
         (metric_logger.acc_ori.global_avg, metric_logger.loss_ori.global_avg,
          metric_logger.loss_train_ori.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/acc', metric_logger.acc.global_avg, steps)
        logger.scalar_summary('eval/loss_test', metric_logger.loss.global_avg, steps)
        logger.scalar_summary('eval/loss_train', metric_logger.loss_train.global_avg, steps)
        logger.scalar_summary('eval/acc_ori', metric_logger.acc_ori.global_avg, steps)
        logger.scalar_summary('eval/loss_test_ori', metric_logger.loss_ori.global_avg, steps)
        logger.scalar_summary('eval/loss_train_ori', metric_logger.loss_train_ori.global_avg, steps)

    model.train(mode)

    return metric_logger.acc.global_avg
