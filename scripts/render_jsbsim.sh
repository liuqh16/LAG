#!/bin/sh
env="SingleControl"     # JSBSim or SingleControl
scenario="heading_task"     # singlecombat_task or heading_task or heading_altitude_task.py
num_agents=1
algo="ppo"
exp="baseline"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" \
    --n-rollout-threads 1 \
    --model-dir "D:\jax\code\CloseAirCombat\envs\JSBSim\model\singlecontrol_baseline.pth"\
    --seed ${seed}