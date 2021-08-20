#!/bin/sh
env="SingleControl"     # JSBSim or SingleControl
task="heading_task"     # singlecombat_task or heading_task or heading_altitude_task.py
num_agents=1
algo="ppo"
exp="heading_0820"
seed=1

echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --task-name ${task} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" --model-dir "results/SingleControl/heading_task/ppo/heading_0820/wandb/latest-run/files"\
    --seed ${seed}