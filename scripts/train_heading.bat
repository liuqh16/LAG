@echo off
set env=SingleControl
set scenario=1/heading
set algo=ppo
set exp=v1
set seed=5

echo env is %env%, scenario is %scenario%, algo is %algo%, exp is %exp%, seed is %seed%

REM 设置 CUDA 设备（仅适用于 NVIDIA 显卡）
set CUDA_VISIBLE_DEVICES=0

REM 运行 Python 训练脚本
python train/train_jsbsim.py ^
    --eval-render-mode real_time --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 ^
    --env-name %env% --algorithm-name %algo% --scenario-name %scenario% --experiment-name %exp% ^
    --seed %seed% --n-training-threads 1 --n-rollout-threads 1 --cuda ^
    --log-interval 1 --save-interval 1 ^
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 ^
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 ^
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8
pause