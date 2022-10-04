#   Copyright (c) 2022 Robert Bosch GmbH
#   Author: Ning Gao
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import numpy as np
import torch
import torch.nn as nn
import pickle
import os


def convert_channel_last_np_to_tensor(input):
    """input: [task_num, samples_num, height, width, channel]"""
    input = torch.from_numpy(input).type(torch.FloatTensor)
    input = input.permute(0, 1, 4, 2, 3).contiguous()
    return input


class ShapeNet1D(object):
    def __init__(self, path, img_size, seed, source, shot, tasks_per_batch, data_size="large"):
        super().__init__()
        self.num_classes = 1
        self.alpha = 0.3
        self.source = source
        self.shot = shot
        self.tasks_per_batch = tasks_per_batch
        self.data_size = data_size
        self.img_size = img_size

        self.x_train, self.y_train = pickle.load(open(os.path.join(path, f"train_data_{data_size}.pkl"), 'rb'))
        self.x_val, self.y_val = pickle.load(open(os.path.join(path, "val_data.pkl"), 'rb'))
        self.x_test, self.y_test = pickle.load(open(os.path.join(path, "test_data.pkl"), 'rb'))

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.y_train = self.y_train[:, :, -1, None]
        self.x_val, self.y_val = np.array(self.x_val), np.array(self.y_val)
        self.y_val = self.y_val[:, :, -1, None]
        self.x_test, self.y_test = np.array(self.x_test), np.array(self.y_test)
        self.y_test = self.y_test[:, :, -1, None]

        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        self.test_counter = 0
        np.random.seed(seed)

    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        """Get data batch."""
        xs, ys, xq, yq = [], [], [], []

        shot = self.shot
        tasks_per_batch = self.tasks_per_batch

        shot_max = shot
        shot = shot_max
        if self.source == 'train':
            x, y = self.x_train, self.y_train
            shot = np.random.randint(3, shot_max + 1)  # context shot is random during training, query num fixed
        elif self.source == 'val':
            x, y = self.x_val, self.y_val
        elif self.source == 'test':
            x, y = self.x_test, self.y_test
        else:
            raise TypeError("no valid dataset type split!")

        for _ in range(tasks_per_batch):
            # sample WAY classes
            classes = np.random.choice(
                range(np.shape(x)[0]), self.num_classes, replace=False)

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []
            for k in list(classes):
                # sample SHOT and QUERY instances
                idx = np.random.choice(
                    range(np.shape(x)[1]),
                    size=shot + shot_max,  # follow the sampling strategy in Pascal1D
                    replace=False)
                x_k = x[k][idx]
                y_k = y[k][idx]

                support_set.append(x_k[:shot])
                query_set.append(x_k[shot:])
                support_sety.append(y_k[:shot])
                query_sety.append(y_k[shot:])

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        xs = np.reshape(
            xs,
            [tasks_per_batch, shot * self.num_classes, *self.img_size])
        xq = np.reshape(
            xq,
            [tasks_per_batch, shot_max * self.num_classes, *self.img_size])

        ys = ys.astype(np.float32) * 2 * np.pi
        yq = yq.astype(np.float32) * 2 * np.pi

        xs = xs.astype(np.float32) / 255.0
        xq = xq.astype(np.float32) / 255.0

        ys = np.concatenate([np.cos(ys), np.sin(ys), ys], axis=-1)
        yq = np.concatenate([np.cos(yq), np.sin(yq), yq], axis=-1)
        xs = convert_channel_last_np_to_tensor(xs)
        xq = convert_channel_last_np_to_tensor(xq)

        batch = {'train': (xs, torch.from_numpy(ys).type(torch.FloatTensor)),
                 'test': (xq, torch.from_numpy(yq).type(torch.FloatTensor))}

        return batch


class AzimuthLoss(nn.Module):
    def __init__(self):
        super(AzimuthLoss, self).__init__()

    def forward(self, outputs, labels):
        return torch.mean(torch.sum((labels[..., :2] - outputs) ** 2, dim=-1))


def degree_loss(outputs, labels):
    labels = torch.rad2deg(labels[..., -1])
    pr_cos = outputs[..., 0]
    pr_sin = outputs[..., 1]
    ps_sin = torch.where(pr_sin >= 0)
    ng_sin = torch.where(pr_sin < 0)
    pr_deg = torch.acos(pr_cos)
    pr_deg_ng = -torch.acos(pr_cos) + 2 * math.pi
    pr_deg[ng_sin] = pr_deg_ng[ng_sin]
    pr_deg = torch.rad2deg(pr_deg)
    errors = torch.stack(
        (torch.abs(labels - pr_deg), torch.abs(labels + 360.0 - pr_deg), torch.abs(labels - (pr_deg + 360.0))), dim=-1)
    errors, _ = torch.min(errors, dim=-1)
    losses = torch.mean(errors)
    return losses