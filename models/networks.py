
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .discriminators import Discriminator


def define_D(input_nc, device):
    # todo
    pass


def define_adder(input_nc, ndf, net, n_layers_G, norm='batch', init_type='normal', init_gain=0.02, device):
    pass


def define_remover(input_nc, ndf, net, n_layers_G, norm='batch', init_type='normal', init_gain=0.02, device):
    pass
