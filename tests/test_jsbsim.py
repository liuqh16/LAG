import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def create_env(config: str):
    env_class = config.split('/')[0]
    if env_class == 'single':
        from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
        return SingleControlEnv(config)
    elif env_class == '1v1':
        from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
        return SingleCombatEnv(config)
    else:
        raise NotImplementedError


class TestJSBSimEnv:

    def test_singlecontrol_env(self):
        # Env Settings test
        env = create_env("single/heading")
        assert env.num_agents == 1
        assert len(env.agents[env.agent_ids[0]].partners) == 0
        assert len(env.agents[env.agent_ids[0]].enemies) == 0

        # DataType test
        #   obs: np.ndarray
        #   action: np.ndarray
        #   rewards: float
        #   done: bool
        #   info: dict
        env.seed(0)
        env.action_space.seed(0)
        obs = env.reset()
        assert isinstance(obs, np.ndarray) \
            and obs.shape == env.observation_space.shape

        obs_buf = [np.copy(obs)]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            act_buf.append(np.copy(action))
            obs_buf.append(np.copy(obs))
            rew_buf.append(reward)
            done_buff.append(done)
            assert isinstance(obs, np.ndarray) \
                and obs.shape == env.observation_space.shape
            assert isinstance(reward, float) \
                and isinstance(done, bool)
            if done:
                assert env.current_step <= env.max_steps
                break

        # Repetition test
        for _ in range(3):
            env.seed(0)
            env.action_space.seed(0)
            obs = env.reset()
            t = 0
            assert np.linalg.norm(obs - obs_buf[t]) < 1e-8
            env.action_space.seed(0)
            while t < len(done_buff):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                assert np.linalg.norm(action - act_buf[t]) < 1e-8 \
                    and np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                    and reward == rew_buf[t] \
                    and done == done_buff[t]
                t += 1

    def test_singlecombat_env(self):
        # Env Settings test
        env = create_env("1v1/NoWeapon/Selfplay")
        assert env.num_agents == 2
        for agent_id in env.agent_ids:
            assert len(env.agents[agent_id].partners) == 0
            assert len(env.agents[agent_id].enemies) == 1

        observation_space = env.observation_space
        action_space = env.action_space
        import pdb; pdb.set_trace()

        # DataType test
        #   obs: np.ndarray
        #   action: np.ndarray
        #   rewards: float
        #   done: bool
        #   info: dict

