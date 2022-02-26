# python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 50000 --num-steps 2000 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 3000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 200 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 1 --log-interval 100 --interaction-probability 0 --max-steps 3000 --pretest-trials 5 --test-trials 5 --temporal-extend 300 --interaction-prediction 0 --breakout-variant center_large --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --test-episode --max-critic 150 --log-interval 25 --save-graph data/breakout/center_large4 --save-interval 100 > logs/breakout/variant_trials/center/train_center_large4.txt
# python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 50000 --num-steps 2000 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 3000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 200 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 1 --log-interval 100 --interaction-probability 0 --max-steps 3000 --pretest-trials 5 --test-trials 5 --temporal-extend 300 --interaction-prediction 0 --breakout-variant center_large --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --test-episode --max-critic 150 --log-interval 25 --save-graph data/breakout/center_large5 --save-interval 100 > logs/breakout/variant_trials/center/train_center_large5.txt
# python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 50000 --num-steps 2000 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 3000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 200 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 1 --log-interval 100 --interaction-probability 0 --max-steps 3000 --pretest-trials 5 --test-trials 5 --temporal-extend 300 --interaction-prediction 0 --breakout-variant center_large --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --test-episode --max-critic 150 --log-interval 25 --save-graph data/breakout/center_large6 --save-interval 100 > logs/breakout/variant_trials/center/train_center_large6.txt

python3 train_offline_baseline.py --algorithm rainbow --observation-type multi-block-encoding --variant breakout_priority_large --seed 567 --epoch 50 --video-log-period 5 --step-per-epoch 100000 --no-render --resume-path data/policy.pth --device cuda:1