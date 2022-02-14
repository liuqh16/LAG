#!/bin/sh
env="SingleCombat"
scenario="1v1/ShootMissile/HierarchySelfplay_nolimit"   
algo="ppo"
exp="fsp"
seed=2021

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --act-hidden-size "128 128" \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay_nolimit/ppo/fsp/wandb/latest-run/files"\
    --seed ${seed} \
    --use-selfplay --selfplay-algorithm "fsp" --use-eval --render-opponent-index "3124"