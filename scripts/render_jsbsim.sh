#!/bin/sh
env="SingleCombat"
scenario="1v1/ShootMissile/HierarchySelfplay"    
algo="ppo"
exp="penalty_shoot_update"
seed=2021

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --act-hidden-size "128 128" \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/penalty_shoot_update/wandb/run-20220425_210318-2ks5keto/files"\
    --use-selfplay --use-eval\
    --use-prior --render-index "latest" --render-opponent-index "500"