import sys
import os
import pytest
import torch
import random
import numpy as np
from pathlib import Path
from itertools import product
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv


class TestSingleControlEnv:

    def test_env(self):
        # Env Settings test
        env = SingleControlEnv("1/heading")
        assert env.num_agents == 1
        for agent in env.agents.values():
            assert len(agent.partners) == 0
            assert len(agent.enemies) == 0
        action_space = env.action_space

        # DataType test
        obs_shape = (env.num_agents, *env.observation_space.shape)
        act_shape = (env.num_agents, *env.action_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        env.seed(0)
        action_space.seed(0)
        obs = env.reset()
        assert obs.shape == obs_shape

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            actions = np.array([action_space.sample() for _ in range(env.num_agents)])
            obs, reward, done, info = env.step(actions)
            assert obs.shape == obs_shape and actions.shape == act_shape \
                and reward.shape == reward_shape and done.shape == done_shape
            act_buf.append(actions)
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
                and np.all(reward == rew_buf[t]) and np.all(done == done_buff[t])
            t += 1

    @pytest.mark.parametrize("vecenv", [DummyVecEnv, SubprocVecEnv])
    def test_vec_env(self, vecenv):
        parallel_num = 4
        envs = vecenv([lambda: SingleControlEnv("1/heading") for _ in range(parallel_num)])

        # DataType test
        obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
        act_shape = (parallel_num, envs.num_agents, *envs.action_space.shape)
        reward_shape = (parallel_num, envs.num_agents, 1)
        done_shape = (parallel_num, envs.num_agents, 1)

        obss = envs.reset()
        assert obss.shape == obs_shape

        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        while True:
            obss, rewards, dones, infos = envs.step(actions)
            assert obss.shape == obs_shape and actions.shape == act_shape \
                and rewards.shape == reward_shape and dones.shape == done_shape \
                and infos.shape[0] == parallel_num and isinstance(infos[0], dict)
            # terminate if any of the parallel envs has been done
            if np.any(dones):
                break
        envs.close()


class TestSingleCombatEnv:

    @pytest.mark.parametrize("config", ["1v1/NoWeapon/vsBaseline", "1v1/NoWeapon/Selfplay",
                                        "1v1/DodgeMissile/vsBaseline", "1v1/DodgeMissile/Selfplay",
                                        "1v1/DodgeMissile/HierarchyVsBaseline", "1v1/DodgeMissile/HierarchySelfplay"])
    def test_env(self, config):
        # Env Settings test
        env = SingleCombatEnv(config)
        for agent in env.agents.values():
            assert len(agent.partners) == 0
            assert len(agent.enemies) == 1
        action_space = env.action_space

        # DataType test
        obs_shape = (env.num_agents, *env.observation_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        env.seed(0)
        action_space.seed(0)
        obs = env.reset()
        assert obs.shape == obs_shape

        obs_buf = [obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            # Legal action inputs: List / np.ndarray
            if env.current_step % 2 == 0:
                actions = [action_space.sample() for _ in range(env.num_agents)]
            else:
                actions = np.array([action_space.sample() for _ in range(env.num_agents)])
            obs, rewards, dones, info = env.step(actions)
            assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape
            # save previous data
            act_buf.append(actions)
            obs_buf.append(obs)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.all(dones):
                assert env.current_step <= env.max_steps
                break

        # Repetition test (same seed => same data)
        env.seed(0)
        obs = env.reset()
        t = 0
        assert np.linalg.norm(obs - obs_buf[t]) < 1e-8
        while t < len(done_buff):
            obs, rewards, dones, info = env.step(act_buf[t])
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                and np.all(rewards == rew_buf[t]) and np.all(dones == done_buff[t])
            t += 1

    @pytest.mark.parametrize("config", ["1v1/NoWeapon/vsBaseline", "1v1/NoWeapon/Selfplay"])
    def test_agent_crash(self, config):
        # if no weapon, once enemy die, env terminate!
        env = SingleCombatEnv(config)
        env.seed(0)
        obs = env.reset()
        env.agents[env.ego_ids[0]].crash()
        actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])
        obs, rewards, dones, info = env.step(actions)
        assert np.min(rewards) < -100  # crash reward!
        assert np.all(dones)

    @pytest.mark.parametrize("config", ["1v1/DodgeMissile/vsBaseline", "1v1/DodgeMissile/Selfplay"])
    def test_agent_shotdown(self, config):
        # if has weapon, once enemy die, env terminate until no missile warning!
        env = SingleCombatEnv(config)
        env.seed(0)
        obs = env.reset()
        crash_id = env.ego_ids[0]   # ego shotdown
        while True:
            # mannual crash
            if env.current_step == 1:
                from envs.JSBSim.core.simulator import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents[crash_id], env.agents[crash_id].enemies[0], 'C0000'))
                env.agents[crash_id].shotdown()
                crash_obs = obs[0]
            actions = np.array([env.action_space.sample() for _ in range(env.num_agents)])

            obs, rewards, dones, info = env.step(actions)

            if np.all(dones):
                break
            elif env.current_step == 2:
                assert np.min(rewards) < -50  # shot down reward!
            elif env.current_step > 2:
                # ego obs is not changed!
                assert dones[0][0] == True \
                    and np.linalg.norm(obs[0, :9] - crash_obs[:9]) < 1e-8 \
                    and rewards[0][0] == 0.0 \
                    and any([missile.is_alive for missile in env.agents[crash_id].launch_missiles])

    @pytest.mark.parametrize("vecenv, config", list(product(
        [DummyVecEnv, SubprocVecEnv], ["1v1/DodgeMissile/Selfplay", "1v1/DodgeMissile/HierarchyVsBaseline"])))
    def test_vec_env(self, vecenv, config):
        parallel_num = 4
        envs = vecenv([lambda: SingleCombatEnv(config) for _ in range(parallel_num)])

        # DataType test
        obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
        reward_shape = (parallel_num, envs.num_agents, 1)
        done_shape = (parallel_num, envs.num_agents, 1)

        obss = envs.reset()
        assert obss.shape == obs_shape

        # Legal action inputs: List / np.ndarray (first ego, then enm)
        actions = [[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)]
        while True:
            obss, rewards, dones, infos = envs.step(actions)
            # check parallel env's data type
            assert obss.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape \
                and infos.shape[0] == parallel_num and isinstance(infos[0], dict)
            # terminate if any of the parallel envs has been done
            if np.any(np.all(dones, axis=1)):
                break
        envs.close()


class TestJSBSimRunner:

    @pytest.mark.parametrize("args", [
        "--env-name SingleControl --algorithm-name ppo --scenario-name 1/heading",
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/Selfplay --use-selfplay --selfplay-algorithm fsp",
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/vsBaseline",
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/HierarchySelfplay",  # whether to use selfplay is optional
        "--env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/DodgeMissile/HierarchyVsBaseline"])
    def test_training(self, args):
        from scripts.train.train_jsbsim import make_train_env, make_eval_env, parse_args, get_config, Runner
        args += ' --experiment-name pytest --seed 1 --n-training-threads 1 --n-rollout-threads 5 --cuda' \
                ' --log-interval 1 --save-interval 1 --use-eval --eval-interval 1 --eval-episodes 10' \
                ' --num-mini-batch 5 --buffer-size 1000 --num-env-steps 1e4' \
                ' --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3' \
                ' --hidden-size 32 --act-hidden-size 32 --recurrent-hidden-size 32 --recurrent-hidden-layers 1 --data-chunk-length 8'
        args = args.split(' ')
        parser = get_config()
        all_args = parse_args(args, parser)

        # seed
        np.random.seed(all_args.seed)
        random.seed(all_args.seed)
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)

        # cuda
        if all_args.cuda and torch.cuda.is_available():
            device = torch.device("cuda:0")  # use cude mask to control using which GPU
            torch.set_num_threads(all_args.n_training_threads)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            torch.set_num_threads(all_args.n_training_threads)

        # run dir
        run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") \
            / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        assert not all_args.use_wandb
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        # env init
        assert all_args.use_eval
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args)

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "device": device,
            "run_dir": run_dir
        }

        # run experiments
        runner = Runner(config)
        runner.run()

        # post process
        envs.close()


class TestMultipleCombatEnv:

    @pytest.mark.parametrize("config", ["2v2/NoWeapon/Selfplay", "2v2/NoWeapon/HierarchySelfplay",
                                        "2v2/ShootMissile/HierarchySelfplay"])
    def test_env(self, config):
        # Env Settings test
        env = MultipleCombatEnv(config)
        assert env.num_agents == 4
        for agent in env.agents.values():
            assert len(agent.partners) == 1
            assert len(agent.enemies) == 2

        obs_shape = (env.num_agents, *env.observation_space.shape)
        share_obs_shape = (env.num_agents, *env.share_observation_space.shape)
        reward_shape = (env.num_agents, 1)
        done_shape = (env.num_agents, 1)

        # DataType test
        env.seed(0)
        env.action_space.seed(0)
        obs, share_obs = env.reset()
        assert obs.shape == obs_shape and share_obs.shape == share_obs_shape

        obs_buf = [obs]
        share_buf = [share_obs]
        act_buf = []
        rew_buf = []
        done_buff = []
        while True:
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
            obs, share_obs, rewards, dones, info = env.step(actions)
            assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
            # save previous data
            obs_buf.append(obs)
            share_buf.append(share_obs)
            act_buf.append(actions)
            rew_buf.append(rewards)
            done_buff.append(dones)
            if np.any(dones[0]):
                assert env.current_step <= env.max_steps
                break

        # Repetition test (same seed => same data)
        env.seed(0)
        obs, share_obs = env.reset()
        t = 0
        assert np.linalg.norm(obs - obs_buf[t]) < 1e-8 \
            and np.linalg.norm(share_obs - share_buf[t]) < 1e-8
        while t < len(done_buff):
            obs, share_obs, rewards, dones, info = env.step(act_buf[t])
            assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
                and np.linalg.norm(share_obs - share_buf[t+1]) < 1e-8 \
                and np.all(rewards == rew_buf[t]) and np.all(dones == done_buff[t])
            t += 1

    def test_agent_die(self):
        env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
        uid = list(env.agents.keys())[0]
        partner_id = env.agents[uid].partners[0].uid
        enemy0_id = env.agents[uid].enemies[0].uid
        enemy1_id = env.agents[uid].enemies[1].uid
        env.seed(0)
        env.reset()
        while True:
            actions = [np.array([20, 18.6, 20, 0]) for _ in range(env.num_agents)]
            if env.current_step == 20:
                env.agents[partner_id].crash()
            if env.current_step == 40:
                env.agents[enemy0_id].crash()
            if env.current_step == 60:
                env.agents[enemy1_id].crash()

            obs, share_obs, rewards, dones, info = env.step(actions)
            rewards = env._unpack(rewards)
            dones = env._unpack(dones)
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
            actions = [np.array([20, 18.6, 20, 0]) for _ in range(env.num_agents)]

            if env.current_step == 20:
                env.agents[enemy0_id].crash()
            if env.current_step == 40:
                env.agents[enemy1_id].crash()
                from envs.JSBSim.core.simulator import MissileSimulator
                env.add_temp_simulator(MissileSimulator.create(env.agents[enemy1_id], env.agents[uid], uid="C0000"))

            obs, share_obs, rewards, dones, info = env.step(actions)
            rewards = env._unpack(rewards)
            dones = env._unpack(dones)
            if env.current_step > 20:
                assert dones[enemy0_id] == True and rewards[enemy0_id] == 0.0
                if env.current_step > 40:
                    assert dones[enemy1_id] == True and rewards[enemy1_id] == 0.0
            if np.all(list(dones.values())):
                assert not env._tempsims["C0000"].is_alive
                break

    @pytest.mark.parametrize("vecenv, config", list(product(
        [ShareDummyVecEnv, ShareSubprocVecEnv], ["2v2/NoWeapon/Selfplay", "2v2/NoWeapon/HierarchySelfplay",
                                        "2v2/ShootMissile/HierarchySelfplay"])))
    def test_vec_env(self, vecenv, config):
        parallel_num = 4
        envs = vecenv([lambda: MultipleCombatEnv(config) for _ in range(parallel_num)])
        assert envs.num_agents == 4
        obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
        share_obs_shape = (parallel_num, envs.num_agents, *envs.share_observation_space.shape)
        reward_shape = (parallel_num, envs.num_agents, 1)
        done_shape = (parallel_num, envs.num_agents, 1)

        # DataType test
        obs, share_obs = envs.reset()
        assert obs.shape == obs_shape and share_obs.shape == share_obs_shape
        while True:
            actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
            obs, share_obs, rewards, dones, info = envs.step(actions)
            assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
            break
        envs.close()
