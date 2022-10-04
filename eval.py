import torch
import torch.nn as nn
from torchmeta.utils.data import BatchMetaDataLoader

from common.args import parse_args
from common.utils import load_model
from data.dataset import get_meta_dataset
from models.model import get_model
from utils import set_random_seed


def main():
    """ argument define """
    P = parse_args()
    P.rank = 0

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    P.world_size = torch.cuda.device_count()
    P.distributed = P.world_size > 1
    assert not P.distributed  # no multi GPU

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    kwargs = {'batch_size': P.test_batch_size, 'shuffle': True,
              'pin_memory': True, 'num_workers': 4}
    test_set = get_meta_dataset(P, dataset=P.dataset, only_test=True)
    if P.regression:
        test_loader = test_set
    else:
        test_loader = BatchMetaDataLoader(test_set, **kwargs)

    """ Initialize model """
    model = get_model(P, P.model).to(device)

    """ load model if necessary """
    load_model(P, model)

    """ define train and test type """
    from evals import setup as test_setup
    test_func = test_setup(P.mode, P)

    """ define loss function """
    if P.dataset == 'pose':
        criterion = nn.MSELoss()
    elif P.dataset == 'shapenet':
        from data.shapenet1d import AzimuthLoss
        criterion = AzimuthLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    """ test """
    test_func(P, model, test_loader, criterion, 0.0, logger=None)


if __name__ == "__main__":
    main()
