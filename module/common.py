"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            initializer.default_weight_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class HyperNetwork_FC(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x

