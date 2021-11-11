#!/bin/sh
env="SingleCombat"     # JSBSim or SingleControl
scenario="1v1/Missile/Selfplay"     # singlecombat_task or heading_task or heading_altitude_task.py
num_agents=2
algo="ppo"
exp="render"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --n-rollout-threads 1 \
    --model-dir "D:\codelib\CloseAirCombat\scripts\results\SingleCombat\1v1\Missile\Selfplay\ppo\test\run1\models\97632000"\
    --seed ${seed}