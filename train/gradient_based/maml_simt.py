import time

import torch
import torch.nn.functional as F

from train.gradient_based import maml_inner_adapt
from train import copy_model_param, param_ema, dropout_eval
from evals import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    filename_with_today_date = True
    assert P.num_shots_global == 0
    return filename_with_today_date


def function_consistency(P, outputs, outputs_target, temp):
    if P.regression:
        return F.mse_loss(outputs, outputs_target)
    p_s = F.log_softmax(outputs / temp, dim=1)
    p_t = F.softmax(outputs_target / temp, dim=1)
    return F.kl_div(p_s, p_t, size_average=False) * (temp ** 2) / outputs.shape[0]


def maml_step(P, step, model, criterion, optimizer, batch, metric_logger, logger):

    stime = time.time()
    inner_loss = 0.
    acc = 0.
    outer_loss = torch.tensor(0., device=device)
    reg_loss = torch.tensor(0., device=device)
    num_tasks = batch['test'][1].size(0)

    if not hasattr(P, 'moving_average'):
        P.moving_average = copy_model_param(model)
    if 'metasgd' in P.mode and not hasattr(P, 'moving_inner_lr'):
        P.moving_inner_lr = copy_model_param(None, params=P.inner_lr)

    if hasattr(P, 'moving_inner_lr'):
        inner_lr_ema = P.moving_inner_lr
    else:
        inner_lr_ema = P.inner_lr

    for task_idx, (train_input, train_target, test_input, test_target) \
            in enumerate(zip(*batch['train'], *batch['test'])):
        train_input = train_input.to(device, non_blocking=True)
        train_target = train_target.to(device, non_blocking=True)
        test_input = test_input.to(device, non_blocking=True)
        test_target = test_target.to(device, non_blocking=True)

        model.train()
        dropout_eval(model)
        params_teacher, _ = maml_inner_adapt(
            model, criterion, train_input, train_target, inner_lr_ema, P.inner_steps,
            first_order=True, params=P.moving_average
        )
        with torch.no_grad():
            outputs_target = model(test_input, params=params_teacher)

        model.train()
        dropout_eval(model)  # do not apply dropout during inner step
        params, loss_train = maml_inner_adapt(
            model, criterion, train_input, train_target, P.inner_lr, P.inner_steps
        )

        """ outer loss aggregate """
        model.train()  # apply dropout at the outer step
        outputs_test = model(test_input, params=params)  # apply dropout
        loss_test = (1. - P.lam) * criterion(outputs_test, test_target)
        loss_reg = P.lam * function_consistency(P, outputs_test, outputs_target, P.temp)

        inner_loss += loss_train.item() / num_tasks
        outer_loss += loss_test / num_tasks
        reg_loss += loss_reg / num_tasks
        if not P.regression:
            acc += accuracy(outputs_test, test_target, topk=(1,))[0].item() / num_tasks

    loss = outer_loss + reg_loss

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    """ exponential weight average """
    param_ema(P, model)

    """ track stat """
    metric_logger.meters['batch_time'].update(time.time() - stime)
    metric_logger.meters['meta_train_cls'].update(inner_loss)
    metric_logger.meters['meta_test_cls'].update(outer_loss.item())
    metric_logger.meters['meta_reg_loss'].update(reg_loss.item())
    metric_logger.meters['train_acc'].update(acc)

    if step % P.print_step == 0:
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary('train/meta_train_cls',
                              metric_logger.meta_train_cls.global_avg, step)
        logger.scalar_summary('train/meta_test_cls',
                              metric_logger.meta_test_cls.global_avg, step)
        logger.scalar_summary('train/meta_reg_loss',
                              metric_logger.meta_reg_loss.global_avg, step)
        logger.scalar_summary('train/train_acc',
                              metric_logger.train_acc.global_avg, step)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, step)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[MetaTrainLoss %f] [MetaTestLoss %f] [MetaRegLoss %f]' %
                   (step, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.meta_train_cls.global_avg, metric_logger.meta_test_cls.global_avg,
                    metric_logger.meta_reg_loss.global_avg))
