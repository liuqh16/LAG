#!/bin/sh
env="MultipleCombat"
scenario="2v2/NoWeapon/Selfplay"    
algo="ppo"
exp="new_v2"
seed=2021

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --act-hidden-size "" \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleCombat/1v1/Missile/HierarchyVsBaseline/ppo/new_v2/wandb/latest-run/files"\
    --seed ${seed}