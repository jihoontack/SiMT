import time

import torch

from train.gradient_based import maml_inner_adapt
from evals import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    filename_with_today_date = True
    assert P.num_shots_global == 0
    return filename_with_today_date


def maml_step(P, step, model, criterion, optimizer, batch, metric_logger, logger):

    stime = time.time()
    model.train()

    inner_loss = 0.
    acc = 0.
    outer_loss = torch.tensor(0., device=device)
    num_tasks = batch['test'][1].size(0)

    for task_idx, (train_input, train_target, test_input, test_target) \
            in enumerate(zip(*batch['train'], *batch['test'])):

        train_input = train_input.to(device, non_blocking=True)
        train_target = train_target.to(device, non_blocking=True)
        test_input = test_input.to(device, non_blocking=True)
        test_target = test_target.to(device, non_blocking=True)

        params, loss_train = maml_inner_adapt(
            model, criterion, train_input, train_target, P.inner_lr, P.inner_steps
        )

        """ outer loss aggregate """
        outputs_test = model(test_input, params=params)
        loss_test = criterion(outputs_test, test_target)

        inner_loss += loss_train.item() / num_tasks
        outer_loss += loss_test / num_tasks
        if not P.regression:
            acc += accuracy(outputs_test, test_target, topk=(1,))[0].item() / num_tasks

    loss = outer_loss

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    """ track stat """
    metric_logger.meters['batch_time'].update(time.time() - stime)
    metric_logger.meters['meta_train_cls'].update(inner_loss)
    metric_logger.meters['meta_test_cls'].update(outer_loss.item())
    metric_logger.meters['train_acc'].update(acc)

    if step % P.print_step == 0:
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary('train/meta_train_cls',
                              metric_logger.meta_train_cls.global_avg, step)
        logger.scalar_summary('train/meta_test_cls',
                              metric_logger.meta_test_cls.global_avg, step)
        logger.scalar_summary('train/train_acc',
                              metric_logger.train_acc.global_avg, step)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, step)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[MetaTrainLoss %f] [MetaTestLoss %f]' %
                   (step, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.meta_train_cls.global_avg, metric_logger.meta_test_cls.global_avg))
