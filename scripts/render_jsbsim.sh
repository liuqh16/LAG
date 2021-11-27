#!/bin/sh
env="SingleControl"     # JSBSim or SingleControl
scenario="single/heading"     # singlecombat_task or heading_task or heading_altitude_task.py
num_agents=1
algo="ppo"
exp="heading_oet"
seed=2

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" \
    --n-rollout-threads 1 \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleControl/single/heading/ppo/heading_norandom/wandb/latest-run/files/90201600"\
    --seed ${seed}