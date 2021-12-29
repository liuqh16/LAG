#!/bin/sh
env="SingleCombat"
scenario="1v1/ShootMissile/HierarchySelfplay"
algo="ppo"
exp="simple_selfplay"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed 1 --n-training-threads 1 --n-rollout-threads 32 --cuda --log-interval 1 --save-interval 1 \
    --use-eval --n-eval-threads 20 --eval-interval 5 --eval-episodes 100 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8
