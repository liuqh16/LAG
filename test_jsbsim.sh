#!/bin/sh
env="heading"
task="heading"
exp="test"
modelpath="results/heading_heading/OriginReward_NotAccReward/models/agent0_history1068.pt"
seed=1

echo "env is ${env}, task is ${scenario}, exp is ${exp}, seed is ${seed}"
python try_test_selfplay.py --env ${env} --task ${task} --exp ${exp} --seed ${seed} --modelpath ${modelpath}
