#!/bin/sh
env="JSBSim"
task="singlecombat"
exp="test"
modelpath="results/JSBSim_selfplay_v0/models/agent0_history0.pt"
seed=1

echo "env is ${env}, task is ${scenario}, exp is ${exp}, seed is ${seed}"
python try_test_jsbsim.py --env ${env} --task ${task} --exp ${exp} --seed ${seed} --modelpath ${modelpath}
