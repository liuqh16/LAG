#!/bin/sh
env="SingleControl"
scenario="single/heading"     
num_agents=1
algo="ppo"
exp="mps_rpsincos_noturnr"
seed=2

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" \
    --n-rollout-threads 1 \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleControl/single/heading/ppo/mps_rpsincos_noturnr/wandb/latest-run/files/99964800"\
    --seed ${seed}