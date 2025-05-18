import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.JSBSim.envs import SingleControlEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

def _t2n(x):
    return x.detach().cpu().numpy()

# === CONFIG ===
render = True
policy_index = "latest"
run_dir = "../scripts/results/SingleControl/1/heading/ppo/v1/wandb/run-20250516_043525-itylszu0/files"
experiment_name = run_dir.split('/')[-4]  
env = SingleControlEnv("1/heading") 
env.seed(0)

# === POLICY ===
args = Args()
policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
policy.eval()
policy.load_state_dict(torch.load(f"{run_dir}/actor_{policy_index}.pt"))

# === INITIALIZE ENV ===
obs = env.reset()
if render:
    env.render(mode='txt', filepath=f"{experiment_name}.txt.acmi")

    # Render static base(s) just once after the ACMI file is created
    static_bases = env.static_bases
    for uid, base in static_bases.items():
        print("->>>>>>>>>>>>>>Rendering static base:", uid, base.origin)
        position = base.origin
        print("filepath:", f"{experiment_name}.txt.acmi")
        env.render_static_base(uid, position, base.color, base.model, base.type, filepath=f"{experiment_name}.txt.acmi")

rnn_states = np.zeros((1, 1, args.recurrent_hidden_size), dtype=np.float32)
masks = np.ones((1, 1), dtype=np.float32)
episode_reward = 0

# === MAIN LOOP ===
while True:
    action, _, rnn_states = policy(obs, rnn_states, masks, deterministic=True)
    action = _t2n(action)
    rnn_states = _t2n(rnn_states)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if render:
        env.render(mode='txt', filepath=f"{experiment_name}.txt.acmi")
    if done.all():
        print(info)
        break

print("Total Episode Reward:", episode_reward)
