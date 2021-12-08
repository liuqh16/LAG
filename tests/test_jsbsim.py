import sys
import os
import pytest
import numpy as np
from copy import deepcopy

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


class TestSingleControlEnv:

    def test_env(self):
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

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            assert isinstance(obs, np.ndarray) \
                and obs.shape == env.observation_space.shape
            assert isinstance(reward, float) \
                and isinstance(done, bool)
            act_buf.append(action)
            obs_buf.append(obs)
            rew_buf.append(reward)
            done_buff.append(done)
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


class TestSingleCombatEnv:

    def test_env(self):
        # Env Settings test
        env = create_env("1v1/NoWeapon/Selfplay")
        assert env.num_agents == 2
        for agent_id in env.agent_ids:
            assert len(env.agents[agent_id].partners) == 0
            assert len(env.agents[agent_id].enemies) == 1
        assert isinstance(env.observation_space, dict) \
            and isinstance(env.action_space, dict)

        # DataType test
        #   obs: np.ndarray
        #   action: np.ndarray
        #   rewards: float
        #   done: bool
        #   info: dict
        env.seed(0)
        for agent_id in env.agent_ids:
            env.action_space[agent_id].seed(0)
        obs = env.reset()
        for agent_id in env.agent_ids:
            assert isinstance(obs[agent_id], np.ndarray) \
                and obs[agent_id].shape == env.observation_space[agent_id].shape

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            actions = {}
            for agent_id in env.agent_ids:
                actions[agent_id] = env.action_space[agent_id].sample()
            obs, rewards, dones, info = env.step(actions)
            for agent_id in env.agent_ids:
                assert isinstance(obs[agent_id], np.ndarray) \
                    and obs[agent_id].shape == env.observation_space[agent_id].shape
                assert isinstance(rewards[agent_id], float) \
                    and isinstance(dones[agent_id], bool)
            # save previous data
            act_buf.append(actions)
            obs_buf.append(obs)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.all(list(dones.values())):
                assert env.current_step <= env.max_steps
                break

        # Repetition test
        for _ in range(3):
            env.seed(0)
            for agent_id in env.agent_ids:
                env.action_space[agent_id].seed(0)
            obs = env.reset()
            t = 0
            for agent_id in env.agent_ids:
                assert np.linalg.norm(obs[agent_id] - obs_buf[t][agent_id]) < 1e-8
            while t < len(done_buff):
                actions = {}
                for agent_id in env.agent_ids:
                    actions[agent_id] = env.action_space[agent_id].sample()
                obs, rewards, dones, info = env.step(actions)
                for agent_id in env.agent_ids:
                    assert np.linalg.norm(actions[agent_id] - act_buf[t][agent_id]) < 1e-8 \
                        and np.linalg.norm(obs[agent_id] - obs_buf[t + 1][agent_id]) < 1e-8 \
                        and rewards[agent_id] == rew_buf[t][agent_id] \
                        and dones[agent_id] == done_buff[t][agent_id]
                t += 1

    def test_agent_die(self):
        env = create_env("1v1/Missile/Selfplay")
        env.seed(0)
        obs = env.reset()

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        agent_die_flags = dict([(agent_id, False) for agent_id in env.agent_ids])
        while True:
            actions = {}
            for agent_id in env.agent_ids:
                actions[agent_id] = np.array([20, 18.6, 20, 0])

            if env.current_step == 30:
                from envs.JSBSim.core.simulatior import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents["A0100"], env.agents["B0100"], 'A01001'))
            obs, rewards, dones, info = env.step(actions)

            for agent_id in env.agent_ids:
                if agent_die_flags[agent_id]:
                    assert np.linalg.norm(obs[agent_id][:11] - obs_buf[-1][agent_id][:11]) < 1e-8 \
                        and rewards[agent_id] == 0 and dones[agent_id]
                if dones[agent_id]:
                    agent_die_flags[agent_id] = True

            # save previous data
            act_buf.append(actions)
            obs_buf.append(obs)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.all(list(dones.values())):
                assert env.current_step <= env.max_steps
                break
