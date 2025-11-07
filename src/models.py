"""
This module contains the definition of the deep learning models for
statistical downscaling. References to each of the models are provided
in the docstring of each class.

Author: Jose González-Abad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class DeepESD(torch.nn.Module):

    def __init__(self, x_shape: tuple, y_shape: tuple, filters_last_conv: int,
                 device: str, orog_data: np.ndarray=None):

        super(DeepESD, self).__init__()

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.orog_data = orog_data
        self.orog_embed_dim = 128

        if self.orog_data is not None:
            self.orog_data = torch.from_numpy(orog_data).float().to(device)
            if len(self.orog_data.shape) == 1:
                self.orog_data = self.orog_data.unsqueeze(0)

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        input_size_linear = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv
        if self.orog_data is not None:
            self.orog_embed = torch.nn.Linear(in_features=self.orog_data.shape[1],
                                              out_features=self.orog_embed_dim)
            input_size_linear += self.orog_embed_dim

        self.out = torch.nn.Linear(in_features=input_size_linear,
                                    out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.orog_data is not None:
            orog_embed = torch.relu(self.orog_embed(self.orog_data))
            orog_embed = orog_embed.repeat(x.size(0), 1)
            x = torch.cat((x, orog_embed), dim=1)

        out = self.out(x)

        return out