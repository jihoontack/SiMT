def setup(mode, P):
    if mode in ['maml', 'metasgd', 'anil']:
        if P.simt:
            from evals.gradient_based.maml_simt import test_classifier as test_func
        else:
            from evals.gradient_based.maml import test_classifier as test_func

    elif mode == 'protonet':
        if P.simt:
            from evals.metric_based.protonet_simt import test_classifier as test_func
        else:
            from evals.metric_based.protonet import test_classifier as test_func

    else:
        print(f'Warning: current running option, i.e., {mode}, needs evaluation code')
        from evals.gradient_based.maml import test_classifier as test_func

    return test_func


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
