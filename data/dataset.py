import torch
from torchvision import transforms

from torchmeta.datasets import MiniImagenet, TieredImagenet, CUB
from torchmeta.transforms import ClassSplitter, Categorical

from data.cars import CARS
from data.pose import Pascal1D
from data.shapenet1d import ShapeNet1D

DATA_PATH = '/data'


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.
    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def resize_transform(resize_size):
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor()
    ])
    return transform


def get_meta_dataset(P, dataset, only_test=False):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=P.num_shots,
                                      num_test_per_class=P.num_shots_test + P.num_shots_global)
    dataset_transform_test = ClassSplitter(shuffle=True,
                                           num_train_per_class=P.num_shots,
                                           num_test_per_class=P.num_shots_test)

    if 'protonet' in P.mode:
        train_num_ways = P.train_num_ways
    else:
        train_num_ways = P.num_ways

    if dataset == 'miniimagenet':
        transform = resize_transform(84)

        meta_train_dataset = MiniImagenet(DATA_PATH,
                                          transform=transform,
                                          target_transform=Categorical(train_num_ways),
                                          num_classes_per_task=train_num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(DATA_PATH,
                                        transform=transform,
                                        target_transform=Categorical(P.num_ways),
                                        num_classes_per_task=P.num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform_test)
        meta_test_dataset = MiniImagenet(DATA_PATH,
                                         transform=transform,
                                         target_transform=Categorical(P.num_ways),
                                         num_classes_per_task=P.num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform_test)

    elif dataset == 'tieredimagenet':
        transform = resize_transform(84)

        meta_train_dataset = TieredImagenet(DATA_PATH,
                                            transform=transform,
                                            target_transform=Categorical(train_num_ways),
                                            num_classes_per_task=train_num_ways,
                                            meta_train=True,
                                            dataset_transform=dataset_transform,
                                            download=True)
        meta_val_dataset = TieredImagenet(DATA_PATH,
                                          transform=transform,
                                          target_transform=Categorical(P.num_ways),
                                          num_classes_per_task=P.num_ways,
                                          meta_val=True,
                                          dataset_transform=dataset_transform_test)
        meta_test_dataset = TieredImagenet(DATA_PATH,
                                           transform=transform,
                                           target_transform=Categorical(P.num_ways),
                                           num_classes_per_task=P.num_ways,
                                           meta_test=True,
                                           dataset_transform=dataset_transform_test)

    elif dataset == 'cub':
        assert only_test
        transform = transforms.Compose([
            transforms.Resize(int(84 * 1.5)),
            transforms.CenterCrop(84),
            transforms.ToTensor()
        ])

        meta_test_dataset = CUB(DATA_PATH,
                                transform=transform,
                                target_transform=Categorical(P.num_ways),
                                num_classes_per_task=P.num_ways,
                                meta_test=True,
                                dataset_transform=dataset_transform_test)

    elif dataset == 'cars':
        assert only_test
        transform = resize_transform(84)
        meta_test_dataset = CARS(DATA_PATH,
                                 transform=transform,
                                 target_transform=Categorical(P.num_ways),
                                 num_classes_per_task=P.num_ways,
                                 meta_test=True,
                                 dataset_transform=dataset_transform_test)

    elif dataset == 'shapenet':
        P.regression = True
        P.num_ways = 2
        meta_train_dataset = ShapeNet1D(path=f'{DATA_PATH}/ShapeNet1D',
                                        img_size=[128, 128, 1],
                                        seed=P.seed,
                                        source='train',
                                        shot=P.num_shots,
                                        tasks_per_batch=P.batch_size)

        meta_val_dataset = ShapeNet1D(path=f'{DATA_PATH}/ShapeNet1D',
                                      img_size=[128, 128, 1],
                                      seed=P.seed,
                                      source='val',
                                      shot=P.num_shots,
                                      tasks_per_batch=P.batch_size)

        meta_test_dataset = ShapeNet1D(path=f'{DATA_PATH}/ShapeNet1D',
                                       img_size=[128, 128, 1],
                                       seed=P.seed,
                                       source='test',
                                       shot=P.num_shots,
                                       tasks_per_batch=P.batch_size)

    elif dataset == 'pose':
        P.regression = True
        P.num_ways = 1
        meta_train_dataset = Pascal1D(path=f'{DATA_PATH}/Pascal1D',
                                      img_size=[128, 128, 1],
                                      seed=P.seed,
                                      source='train',
                                      shot=P.num_shots,
                                      tasks_per_batch=P.batch_size)

        meta_val_dataset = Pascal1D(path=f'{DATA_PATH}/Pascal1D',
                                    img_size=[128, 128, 1],
                                    seed=P.seed,
                                    source='val',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size)

        meta_test_dataset = meta_val_dataset

    else:
        raise NotImplementedError()

    if only_test:
        return meta_test_dataset

    return meta_train_dataset, meta_val_dataset
