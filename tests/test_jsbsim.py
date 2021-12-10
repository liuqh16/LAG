import sys
import os
import pytest
import numpy as np
from itertools import product
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv


class TestSingleControlEnv:

    def test_env(self):
        # Env Settings test
        env = SingleControlEnv("single/heading")
        assert env.num_agents == 1
        assert len(env.agents[env.agent_ids[0]].partners) == 0
        assert len(env.agents[env.agent_ids[0]].enemies) == 0

        # DataType test
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

        # Repetition test (same seed => same data)
        env.seed(0)
        obs = env.reset()
        t = 0
        assert np.linalg.norm(obs - obs_buf[t]) < 1e-8
        while t < len(done_buff):
            obs, reward, done, info = env.step(act_buf[t])
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                and reward == rew_buf[t] \
                and done == done_buff[t]
            t += 1

    @pytest.mark.parametrize("vecenv", [DummyVecEnv, SubprocVecEnv])
    def test_vec_env(self, vecenv):
        parallel_num = 4
        envs = vecenv([lambda: SingleControlEnv("single/heading") for _ in range(parallel_num)])
        obss = envs.reset()
        assert isinstance(obss, np.ndarray) and obss.shape[0] == parallel_num

        actions = [envs.action_space.sample() for _ in range(parallel_num)]
        while True:
            obss, rewards, dones, infos = envs.step(actions)
            # check parallel env's data type
            assert isinstance(obss, np.ndarray) and obss.shape[0] == parallel_num \
                and isinstance(rewards, np.ndarray) and rewards.shape[0] == parallel_num \
                and isinstance(dones, np.ndarray) and dones.shape[0] == parallel_num \
                and isinstance(infos, np.ndarray) and infos.shape[0] == parallel_num
            # terminate if any of the parallel envs has been done
            if np.any(dones):
                break
        envs.close()


class TestSingleCombatEnv:

    def test_env(self):
        # Env Settings test
        env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
        assert env.num_agents == 2
        for agent_id in env.agent_ids:
            assert len(env.agents[agent_id].partners) == 0
            assert len(env.agents[agent_id].enemies) == 1
        assert isinstance(env.observation_space, dict) \
            and isinstance(env.action_space, dict)

        # DataType test
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

        # Repetition test (same seed => same data)
        env.seed(0)
        obs = env.reset()
        t = 0
        for agent_id in env.agent_ids:
            assert np.linalg.norm(obs[agent_id] - obs_buf[t][agent_id]) < 1e-8
        while t < len(done_buff):
            obs, rewards, dones, info = env.step(act_buf[t])
            for agent_id in env.agent_ids:
                assert np.linalg.norm(obs[agent_id] - obs_buf[t + 1][agent_id]) < 1e-8 \
                    and rewards[agent_id] == rew_buf[t][agent_id] \
                    and dones[agent_id] == done_buff[t][agent_id]
            t += 1

    def test_agent_die(self):
        # if no weapon, once enemy die, env terminate!
        env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
        env.seed(0)
        obs = env.reset()
        env.agents[env.agent_ids[0]].crash()
        actions = dict([(agent_id, np.array([20, 18.6, 20, 0])) for agent_id in env.agent_ids])
        obs, rewards, dones, info = env.step(actions)
        assert np.all(list(dones.values()))

        # if has weapon, once enemy die, env terminate until no missile warning!
        env = SingleCombatEnv("1v1/Missile/Selfplay")
        env.seed(0)
        obs = env.reset()
        crash_id = env.agent_ids[0]
        while True:
            # mannual crash
            if env.current_step == 1:
                from envs.JSBSim.core.simulatior import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents[crash_id], env.agents[crash_id].enemies[0], 'C0000'))
                env.agents[crash_id].crash()
                crash_obs = obs[crash_id]
            actions = dict([(agent_id, np.array([20, 18.6, 20, 0])) for agent_id in env.agent_ids])

            obs, rewards, dones, info = env.step(actions)

            if np.all(list(dones.values())):
                print(info)
                break
            elif env.current_step > 1:
                assert dones[crash_id] == True \
                    and np.linalg.norm(obs[crash_id][:11] - crash_obs[:11]) < 1e-8 \
                    and rewards[crash_id] == 0.0 \
                    and np.any([missile.is_alive for missile in env.agents[crash_id].launch_missiles])

    @pytest.mark.parametrize("vecenv, config", list(product(
        [DummyVecEnv, SubprocVecEnv], ["1v1/NoWeapon/Selfplay", "1v1/Missile/Selfplay"])))
    def test_vec_env(self, vecenv, config):
        parallel_num = 4
        envs = vecenv([lambda: SingleCombatEnv(config) for _ in range(parallel_num)])
        obss = envs.reset()
        assert obss.shape[0] == parallel_num

        actions = [dict([(agent_id, envs.action_space[agent_id].sample()) for agent_id in envs.agent_ids]) for _ in range(parallel_num)]
        while True:
            obss, rewards, dones, infos = envs.step(actions)
            # check parallel env's data type
            assert isinstance(obss, np.ndarray) and isinstance(obss[0], dict) and obss.shape[0] == parallel_num \
                and isinstance(rewards, np.ndarray) and isinstance(rewards[0], dict) and rewards.shape[0] == parallel_num \
                and isinstance(dones, np.ndarray) and isinstance(dones[0], dict) and dones.shape[0] == parallel_num \
                and isinstance(infos, np.ndarray) and isinstance(infos[0], dict) and infos.shape[0] == parallel_num
            for i in range(parallel_num):
                for agent_id in envs.agent_ids:
                    assert obss[i][agent_id].shape == envs.observation_space[agent_id].shape \
                        and isinstance(rewards[i][agent_id], float) \
                        and isinstance(dones[i][agent_id], bool)
            # terminate if any of the parallel envs has been done
            if np.any(list(map(lambda x: np.all(list(x.values())), dones))):
                break
        envs.close()


class TestMultipleCombatEnv:

    def test_env(self):
        # Env Settings test
        env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
        assert env.num_agents == 4
        for agent_id in env.agent_ids:
            assert len(env.agents[agent_id].partners) == 1
            assert len(env.agents[agent_id].enemies) == 2
        assert isinstance(env.observation_space, dict) \
            and isinstance(env.share_observation_space, dict) \
            and isinstance(env.action_space, dict)

        # DataType test
        env.seed(0)
        for agent_id in env.agent_ids:
            env.action_space[agent_id].seed(0)
        obs, share_obs = env.reset()
        for agent_id in env.agent_ids:
            assert isinstance(obs[agent_id], np.ndarray) \
                and obs[agent_id].shape == env.observation_space[agent_id].shape \
                and isinstance(share_obs[agent_id], np.ndarray) \
                and share_obs[agent_id].shape == env.share_observation_space[agent_id].shape

        obs_buf = [obs]
        share_buf = [share_obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            actions = {}
            for agent_id in env.agent_ids:
                actions[agent_id] = env.action_space[agent_id].sample()
            obs, share_obs, rewards, dones, info = env.step(actions)
            for agent_id in env.agent_ids:
                assert isinstance(obs[agent_id], np.ndarray) \
                    and obs[agent_id].shape == env.observation_space[agent_id].shape \
                    and isinstance(share_obs[agent_id], np.ndarray) \
                    and share_obs[agent_id].shape == env.share_observation_space[agent_id].shape \
                    and isinstance(rewards[agent_id], float) \
                    and isinstance(dones[agent_id], bool)
            # save previous data
            obs_buf.append(obs)
            share_buf.append(share_obs)
            act_buf.append(actions)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.all(list(dones.values())):
                assert env.current_step <= env.max_steps
                break

        # Repetition test (same seed => same data)
        env.seed(0)
        obs, share_obs = env.reset()
        t = 0
        for agent_id in env.agent_ids:
            assert np.linalg.norm(obs[agent_id] - obs_buf[t][agent_id]) < 1e-8 \
                and np.linalg.norm(share_obs[agent_id] - share_buf[t][agent_id]) < 1e-8
        while t < len(done_buff):
            obs, share_obs, rewards, dones, info = env.step(act_buf[t])
            for agent_id in env.agent_ids:
                assert np.linalg.norm(obs[agent_id] - obs_buf[t + 1][agent_id]) < 1e-8 \
                    and np.linalg.norm(share_obs[agent_id] - share_buf[t + 1][agent_id]) < 1e-8 \
                    and rewards[agent_id] == rew_buf[t][agent_id] \
                    and dones[agent_id] == done_buff[t][agent_id]
            t += 1

    def test_agent_die(self):
        # if no weapon, once all enemies die, env terminate!
        env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
        partner_id = env.agents[env.agent_ids[0]].partners[0].uid
        enemy0_id = env.agents[env.agent_ids[0]].enemies[0].uid
        enemy1_id = env.agents[env.agent_ids[0]].enemies[1].uid
        env.seed(0)
        env.reset()
        while True:
            actions = {}
            for agent_id in env.agent_ids:
                actions[agent_id] = np.array([20, 18.6, 20, 0])

            if env.current_step == 20:
                env.agents[partner_id].crash()
            if env.current_step == 40:
                env.agents[enemy0_id].crash()
            if env.current_step == 60:
                env.agents[enemy1_id].crash()

            obs, share_obs, rewards, dones, info = env.step(actions)

            if env.current_step > 20:
                assert dones[partner_id] == True and rewards[partner_id] == 0.0
                if env.current_step > 40:
                    assert dones[enemy0_id] == True and rewards[enemy0_id] == 0.0
            if env.current_step == 61:
                assert np.all(list(dones.values()))
                break

        # if has weapon, once all enemies die, env terminate until no missile warning!
        env.seed(0)
        env.reset()
        while True:
            actions = {}
            for agent_id in env.agent_ids:
                actions[agent_id] = np.array([20, 18.6, 20, 0])

            if env.current_step == 20:
                env.agents[enemy0_id].crash()
            if env.current_step == 40:
                env.agents[enemy1_id].crash()
                from envs.JSBSim.core.simulatior import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents[enemy1_id], env.agents[env.agent_ids[0]], uid="C0000"))

            obs, share_obs, rewards, dones, info = env.step(actions)

            if env.current_step > 20:
                assert dones[enemy0_id] == True and rewards[enemy0_id] == 0.0
                if env.current_step > 40:
                    assert dones[enemy1_id] == True and rewards[enemy1_id] == 0.0
            if np.all(list(dones.values())):
                assert not env._tempsims["C0000"].is_alive
                break

    @pytest.mark.parametrize("vecenv, config", list(product(
        [ShareDummyVecEnv, ShareSubprocVecEnv], ["2v2/NoWeapon/Selfplay"])))
    def test_vec_env(self, vecenv, config):
        parallel_num = 4
        envs = vecenv([lambda: MultipleCombatEnv(config) for _ in range(parallel_num)])
        obss, share_obss = envs.reset()
        assert obss.shape[0] == parallel_num and \
            share_obss.shape[0] == parallel_num

        actions = [dict([(agent_id, envs.action_space[agent_id].sample()) for agent_id in envs.agent_ids]) for _ in range(parallel_num)]
        while True:
            obss, share_obss, rewards, dones, infos = envs.step(actions)
            # check parallel env's data type
            assert isinstance(obss, np.ndarray) and isinstance(obss[0], dict) and obss.shape[0] == parallel_num \
                and isinstance(share_obss, np.ndarray) and isinstance(share_obss[0], dict) and share_obss.shape[0] == parallel_num \
                and isinstance(rewards, np.ndarray) and isinstance(rewards[0], dict) and rewards.shape[0] == parallel_num \
                and isinstance(dones, np.ndarray) and isinstance(dones[0], dict) and dones.shape[0] == parallel_num \
                and isinstance(infos, np.ndarray) and isinstance(infos[0], dict) and infos.shape[0] == parallel_num
            for i in range(parallel_num):
                for agent_id in envs.agent_ids:
                    assert obss[i][agent_id].shape == envs.observation_space[agent_id].shape \
                        and isinstance(rewards[i][agent_id], float) \
                        and isinstance(dones[i][agent_id], bool)
            # terminate if any of the parallel envs has been done
            if np.any(list(map(lambda x: np.all(list(x.values())), dones))):
                break
        envs.close()
