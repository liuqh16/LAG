import gym
import numpy as np
from collections import OrderedDict
from envs.collections_task.self_play_task import act_space, obs_space


def get_obs_slice():
    obs_slice = {}
    offset = 0

    # length = obs_space['blue_car']['laser_info'].shape[0]
    # obs_slice['laser_shape'] = (offset, offset + length)
    # offset += length

    length = obs_space['blue_fighter']['ego_info'].shape[0]
    obs_slice['ego_shape'] = (offset, offset + length)
    offset += length

    # length = obs_space['blue_car']['key_info'].shape[0]
    # obs_slice['key_shape'] = (offset, offset + length)
    # offset += length

    return obs_slice


obs_slice = get_obs_slice()


def get_act_slice():
    act_slice = {}
    offset = 0

    length = 1
    act_slice['cmd_id'] = (offset, offset + length)
    offset += length

    length = 4
    act_slice['cmd_param'] = (offset, offset + length)
    offset += length

    length = 1
    act_slice['cmd_shoot'] = (offset, offset + length)
    offset += length

    length = 1
    act_slice['cmd_target'] = (offset, offset + length)
    offset += length

    length = 1
    act_slice['cmd_ew'] = (offset, offset + length)
    offset += length

    return act_slice


act_slice = get_act_slice()


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
    """把Box类型的空间变成一个Vector
    """

    def __init__(self, ori_space):
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Box)
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
    """把Box类型的空间变成一个Vector
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


# if __name__ == "__main__":
#     # 测试字典类型
#     space = gym.spaces.Dict({
#         "A1":
#         gym.spaces.Dict({
#             "B1": gym.spaces.Box(low=0., high=1., shape=(3, 4)),
#             "B2": gym.spaces.Discrete(2),
#         }),
#         "A2":
#         gym.spaces.Box(low=0., high=1., shape=(5,)),
#     })
#     sample = space.sample()
#     print(sample)
#
#     flattener = DictFlattener(space)
#     sample_flattened = flattener(sample)
#     print(sample_flattened)
#     sample_flattenback = flattener.inv(sample_flattened)
#     print(sample_flattenback)
#
#     # 测试动作空间采样
#     action = act_space.sample()
#     flattener = DictFlattener(act_space)
#     action_flattened = flattener(action)
#     assert np.all(action['cmd_id'] == action_flattened[slice(*act_slice['cmd_id'])])
#     assert np.all(action['cmd_param'] == action_flattened[slice(*act_slice['cmd_param'])])
#     assert np.all(action['cmd_shoot'] == action_flattened[slice(*act_slice['cmd_shoot'])])
#     assert np.all(action['cmd_target'] == action_flattened[slice(*act_slice['cmd_target'])])
#     print('Test for action space passed')
#
#     # 测试观测空间采样
#     obs = obs_space.sample()
#     flattener = DictFlattener(obs_space)
#     obs_flattened = flattener(obs)
#     assert np.all(obs['blue_plane']['aa_info'].ravel() == obs_flattened[slice(*obs_slice['blue_aa'])])
#     assert np.all(obs['blue_plane']['fighter_info'] == obs_flattened[slice(*obs_slice['blue_fighter'])])
#     assert np.all(obs['blue_plane']['msl_info'].ravel() == obs_flattened[slice(*obs_slice['blue_msl'])])
#     assert np.all(obs['blue_plane']['state_info'] == obs_flattened[slice(*obs_slice['blue_state'])])
#     assert np.all(obs['blue_plane']['target_info'] == obs_flattened[slice(*obs_slice['blue_target'])])
#     assert np.all(obs['red_plane']['aa_info'].ravel() == obs_flattened[slice(*obs_slice['red_aa'])])
#     assert np.all(obs['red_plane']['fighter_info'] == obs_flattened[slice(*obs_slice['red_fighter'])])
#     assert np.all(obs['red_plane']['msl_info'].ravel() == obs_flattened[slice(*obs_slice['red_msl'])])
#     assert np.all(obs['red_plane']['state_info'] == obs_flattened[slice(*obs_slice['red_state'])])
#     assert np.all(obs['red_plane']['target_info'] == obs_flattened[slice(*obs_slice['red_target'])])
#     print('Test for obs space passed')