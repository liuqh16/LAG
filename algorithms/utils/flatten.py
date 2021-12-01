import gym.spaces
import numpy as np
from collections import OrderedDict


def build_flattener(space):
    if isinstance(space, gym.spaces.Dict):
        return DictFlattener(space)
    elif isinstance(space, gym.spaces.Box) \
            or isinstance(space, gym.spaces.MultiDiscrete):
        return BoxFlattener(space)
    elif isinstance(space, gym.spaces.Discrete):
        return DiscreteFlattener(space)
    else:
        raise NotImplementedError


class DictFlattener():
    """Dict和Vector直接的转换
    """

    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Dict)
        self.size = 0
        self.flatteners = OrderedDict()
        for name, space in self.space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                flattener = BoxFlattener(space)
            elif isinstance(space, gym.spaces.Discrete):
                flattener = DiscreteFlattener(space)
            elif isinstance(space, gym.spaces.Dict):
                flattener = DictFlattener(space)
            self.flatteners[name] = flattener
            self.size += flattener.size

    def __call__(self, observation):
        """把Dict转换成Vector
        """
        assert isinstance(observation, OrderedDict)
        batch = self.get_batch(observation, self)
        if batch == 1:
            array = np.zeros(self.size,)
        else:
            array = np.zeros(self.size)

        self.write(observation, array, 0)
        return array

    def inv(self, observation):
        """把Vector解码成Dict
        """
        offset_start, offset_end = 0, 0
        output = OrderedDict()
        for n, f in self.flatteners.items():
            offset_end += f.size
            output[n] = f.inv(observation[..., offset_start:offset_end])
            offset_start = offset_end
        return output

    def write(self, observation, array, offset):
        for o, f in zip(observation.values(), self.flatteners.values()):
            f.write(o, array, offset)
            offset += f.size

    def get_batch(self, observation, flattener):
        if isinstance(observation, dict):
            # 如果是字典的话返回第一个的batch
            for o, f in zip(observation.values(), flattener.flatteners.values()):
                return self.get_batch(o, f)
        else:
            return np.asarray(observation).size // flattener.size


class BoxFlattener():
    """把Box/MultiDiscrete类型的空间变成一个Vector
    """

    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Box) \
            or isinstance(ori_space, gym.spaces.MultiDiscrete)
        self.size = np.product(ori_space.shape)

    def __call__(self, observation):
        array = np.array(observation, copy=False)
        if array.size // self.size == 1:
            return array.ravel()
        else:
            return array.reshape(-1, self.size)

    def inv(self, observation):
        array = np.array(observation, copy=False)
        if array.size // self.size == 1:
            return array.reshape(self.space.shape)
        else:
            return array.reshape((-1,) + self.space.shape)

    def write(self, observation, array, offset):
        array[..., offset:offset + self.size] = self(observation)


class DiscreteFlattener():
    """把Discrete类型的空间变成一个Vector
    """

    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Discrete)
        self.size = 1

    def __call__(self, observation):
        array = np.array(observation, copy=False)
        if array.size == 1:
            return array.item()
        else:
            return array.reshape(-1, 1)

    def inv(self, observation):
        array = np.array(observation, dtype=np.int, copy=False)
        if array.size == 1:
            return array.item()
        else:
            return array.reshape(-1, 1)

    def write(self, observation, array, offset):
        array[..., offset:offset + 1] = self(observation)
