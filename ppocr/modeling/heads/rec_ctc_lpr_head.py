from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle.nn import functional as F


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


class CTCLPRHeadV1(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.container = conv2d(in_channels=in_channels,
                                out_channels=self.out_channels,
                                kernel_size=(1, 1),
                                stride=(1, 1))

    def forward(self, x, targets=None):
        x = self.container(x)
        x = paddle.mean(x, axis=2)
        predicts = paddle.transpose(x, perm=[0, 2, 1])

        if not self.training:
            predicts = F.softmax(predicts, axis=2)

        result = predicts

        return result
