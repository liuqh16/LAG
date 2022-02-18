#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=3
num_agents=3
algo="mappo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --env-name ${env} --algorithm-name ${algo} --experiment-name ${exp} --scenario-name ${scenario} --num-agents ${num_agents} --num-landmarks ${num_landmarks} --seed ${seed} --n-training-threads 1 --n-rollout-threads 128 --num-mini-batch 1 --episode-length 25 --num-env-steps 20000000 --ppo-epoch 10 --gain 0.01 --lr 7e-4 --user-name "jyh" --cuda
done