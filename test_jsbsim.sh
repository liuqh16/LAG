#!/bin/sh
env="JSBSim"
task="selfplay"
exp="test"
modelpath="results/JSBSim_selfplay/VsBaseline_OriginReward/models/agent0_history900.pt"
seed=1

echo "env is ${env}, task is ${scenario}, exp is ${exp}, seed is ${seed}"
python try_test_selfplay.py --env ${env} --task ${task} --exp ${exp} --seed ${seed} --modelpath ${modelpath}
