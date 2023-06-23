from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0):
    return nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
        bias_attr=False)


class small_basic_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        squeezed_ch = ch_out // 4
        super().__init__()
        self.block = nn.Sequential(
            conv2d(in_channels=ch_in,
                   out_channels=squeezed_ch,
                   kernel_size=1),
            nn.ReLU(),
            conv2d(ch_out // 4,
                   ch_out // 4,
                   kernel_size=(3, 1),
                   padding=(1, 0)),
            nn.ReLU(),
            conv2d(ch_out // 4,
                   ch_out // 4,
                   kernel_size=(1, 3),
                   padding=(0, 1)),
            nn.ReLU(),
            conv2d(ch_out // 4,
                   ch_out,
                   kernel_size=1),
        )
    
    def forward(self, x):
        return self.block(x)


class LPRNetPaddleV1(nn.Layer):
    """
    Implementation of Pytorch LPRNetEnhanceV20.
    """

    def __init__(self,
                 in_channels,
                 dropout_rate,
                 class_num):
        super().__init__()
        assert isinstance(in_channels, int)

        self.class_num = class_num

        self._maxpool2d_1 = nn.MaxPool2D(3, stride=1)
        self._maxpool2d_2 = nn.MaxPool2D(3, stride=(1, 2))
        self._maxpool2d_3 = nn.MaxPool2D(3, stride=(1, 2), padding=(0, 1))
        self._stem = nn.Sequential(
            conv2d(in_channels=in_channels,
                   out_channels=16,
                   kernel_size=3,
                   stride=(1, 2),
                   padding=1),
            nn.ReLU(),
            conv2d(in_channels=16,
                   out_channels=64,
                   kernel_size=3,
                   stride=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )
        self._stage_1 = nn.Sequential(
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2D(num_features=128),
            nn.ReLU()
        )
        self._stage_2 = nn.Sequential(
            small_basic_block(ch_in=128, ch_out=256),
            nn.BatchNorm2D(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2D(num_features=256),
            nn.ReLU()
        )
        self._stage_3 = nn.Sequential(
            conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            conv2d(in_channels=64,
                   out_channels=256,
                   kernel_size=(1, 5),
                   stride=1), 
            nn.BatchNorm2D(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            conv2d(in_channels=256,
                   out_channels=class_num,
                   kernel_size=(13, 1),
                   stride=1),
            nn.BatchNorm2D(num_features=class_num),
            nn.ReLU()
        )
        self.out_channels = int(448+class_num)

    def forward(self, x):
        keep_features = list()

        x = self._stem(x)
        keep_features.append(x)
        x = self._maxpool2d_1(x)

        x = self._stage_1(x)
        keep_features.append(x)
        x = self._maxpool2d_2(x)

        x = self._stage_2(x)
        keep_features.append(x)
        x = self._maxpool2d_3(x)

        x = self._stage_3(x)
        keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2D(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2D(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = paddle.pow(f, 2)
            f_mean = paddle.mean(
                paddle.mean(paddle.mean(f_pow, axis=-1), axis=-1), axis=-1
            )
            f_mean = paddle.unsqueeze(f_mean, 1)
            f_mean = paddle.unsqueeze(f_mean, 1)
            f_mean = paddle.unsqueeze(f_mean, 1)
            f = paddle.divide(f, f_mean)
            global_context.append(f)

        x = paddle.concat(global_context, 1)

        return x
