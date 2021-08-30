#!/bin/sh
env="JSBSim"     # JSBSim or SingleControl
scenario="singlecombat_with_artillery_selfplay"     # singlecombat_task or heading_task or heading_altitude_task.py
num_agents=2
algo="ppo"
exp="SelfPlay"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" \
    --n-rollout-threads 1 \
    --model-dir "/home/lqh/jyh/CloseAirCombat/scripts/results/JSBSim/singlecombat_with_artillery_selfplay/ppo/SelfPlay/wandb/latest-run/files"\
    --seed ${seed}