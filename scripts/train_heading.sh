#!/bin/sh
env="SingleControl"
scenario="1/heading"
algo="ppo"
exp="v1"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 1 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 1 --buffer-size 128 --num-env-steps 1024 \
    --lr 0.01 --ppo-epoch 1 --clip-params 0.2 --max-grad-norm 1 --entropy-coef 0.0 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --user-name "jyh" --wandb-name "jsbsim_test_run" --use-wandb