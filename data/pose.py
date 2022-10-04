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

import numpy as np
import torch
import pickle
import os


def convert_channel_last_np_to_tensor(input):
    """input: [task_num, samples_num, height, width, channel]"""
    input = torch.from_numpy(input).type(torch.FloatTensor)
    input = input.permute(0, 1, 4, 2, 3).contiguous()
    return input


class Pascal1D(object):
    def __init__(self, path, img_size, seed, source, shot, tasks_per_batch):
        super().__init__()
        self.num_classes = 1
        self.img_size = img_size
        self.alpha = 0.3
        self.source = source
        self.shot = shot
        self.tasks_per_batch = tasks_per_batch

        self.x_train, self.y_train = pickle.load(open(os.path.join(path, "train_data_ins.pkl"), 'rb'))
        self.x_val, self.y_val = pickle.load(open(os.path.join(path, "val_data_ins.pkl"), 'rb'))

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.y_train = self.y_train[:, :, -1, None]
        self.x_val, self.y_val = np.array(self.x_val), np.array(self.y_val)
        self.y_val = self.y_val[:, :, -1, None]

        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)

    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        """Get data batch."""
        xs, ys, xq, yq = [], [], [], []
        if self.source == 'train':
            x, y = self.x_train, self.y_train
        elif self.source == 'val':
            x, y = self.x_val, self.y_val
        else:
            raise NotImplementedError()

        for _ in range(self.tasks_per_batch):
            # sample tasks
            classes = np.random.choice(
                range(np.shape(x)[0]), self.num_classes, replace=False)

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []
            for k in list(classes):
                idx = np.random.choice(
                    range(np.shape(x)[1]),
                    size=self.shot + self.shot,
                    replace=False)
                x_k = x[k][idx]
                y_k = y[k][idx]

                support_set.append(x_k[:self.shot])
                query_set.append(x_k[self.shot:])
                support_sety.append(y_k[:self.shot])
                query_sety.append(y_k[self.shot:])

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
            [self.tasks_per_batch, self.shot * self.num_classes, *self.img_size])
        xq = np.reshape(
            xq,
            [self.tasks_per_batch, self.shot * self.num_classes, *self.img_size])

        xs = xs.astype(np.float32) / 255.0
        xq = xq.astype(np.float32) / 255.0
        ys = ys.astype(np.float32) * 10.0
        yq = yq.astype(np.float32) * 10.0

        xs = convert_channel_last_np_to_tensor(xs)
        xq = convert_channel_last_np_to_tensor(xq)

        batch = {'train': (xs, torch.from_numpy(ys).type(torch.FloatTensor)),
                 'test': (xq, torch.from_numpy(yq).type(torch.FloatTensor))}

        return batch
