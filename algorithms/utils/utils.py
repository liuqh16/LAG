import copy
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_shape_from_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(space, gym.spaces.Box) \
            or isinstance(space, gym.spaces.MultiDiscrete) \
            or isinstance(space, gym.spaces.MultiBinary):
        return space.shape
    elif isinstance(space,gym.spaces.Tuple) and \
           isinstance(space[0], gym.spaces.MultiDiscrete) and \
               isinstance(space[1], gym.spaces.Discrete):
        return (space[0].shape[0] + 1,)
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(space)}!")


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def init(module: nn.Module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
