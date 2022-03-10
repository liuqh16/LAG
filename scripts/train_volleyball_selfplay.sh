env="gym"
scenario="Volleyball-v0"
algo="ppo"
exp="pfsp"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=2 python train/train_gym.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda --log-interval 1 --save-interval 5 \
    --num-mini-batch 5 --buffer-size 6000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-selfplay --selfplay-algorithm "pfsp" --n-choose-opponents 4 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 10 --eval-episodes 4 \
    # --user-name "jyh" --use-wandb --wandb-name "jyh"