#!/bin/sh
env="SingleCombat"
scenario="1v1/Missile/HierarchyVsBaseline"    
num_agents=1
algo="ppo"
exp="opposite_15km_attack12000"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --n-rollout-threads 1 \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleCombat/1v1/Missile/HierarchyVsBaseline/ppo/opposite_15km_attack12000/wandb/latest-run/files/95126400"\
    --seed ${seed}