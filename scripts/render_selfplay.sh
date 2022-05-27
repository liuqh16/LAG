#!/bin/sh
env="MultipleCombat"
scenario="2v2/NoWeapon/HierarchySelfplay"   
algo="mappo"
exp="fsp"
seed=2021

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --act-hidden-size "128 128" \
    --model-dir "/home/jjh/code/CloseAirCombat/scripts/results/MultipleCombat/2v2/NoWeapon/HierarchySelfplay/mappo/check/wandb/latest-run/files"\
    --seed ${seed} \
    --use-selfplay --selfplay-algorithm "fsp" --use-eval \
    --render-index "latest"  --render-opponent-index "latest"