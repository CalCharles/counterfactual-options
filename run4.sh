# hindsight limit
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 0 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 30 --param-interaction > logs/param_tests/train_ball_velocitymh30.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 0 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction > logs/param_tests/train_ball_velocitymh20.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 0 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 10 --param-interaction > logs/param_tests/train_ball_velocitymh10.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 1000 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 0 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha 1 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocitysac1.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 1000 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 0 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .1 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocitysacp1.txt

# true interactions
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --true-interaction > logs/breakout/hyper/bvti1-100--1.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --true-interaction > logs/breakout/hyper/bvti1-200--1.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --true-interaction > logs/breakout/hyper/bvti100-200--1.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 10 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -.01 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --true-interaction > logs/breakout/hyper/bvti1-10--01.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 0 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -.01 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --true-interaction > logs/breakout/hyper/bvti0-100--01.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 3000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -.1 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 1 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --true-interaction --save-graph data/breakout/hypertest/bvit100-200--p1t1 --save-interval 100 > logs/breakout/hyper/bvit100-200--p1t1.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 5000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.2 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 2 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bvit1-200--1te2.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 1000000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 3000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 3 1 --her-only-interact 1 --resample-timer 0 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --env-reset --sum-rewards --save-graph data/breakout/hyper/ball_graphsum --save-interval 100 > logs/breakout/hyper/train_ballsum.txt

# only use relative state and param
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 1000000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 20000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 3 --hardcode-norm breakout 3 1 --her-only-interact 1 --resample-timer 0 --drop-stopping --observation-setting 0 0 0 1 0 1 0 0 0 --env-reset --sample-merged --save-graph data/breakout/hyper/ball_graphrel --save-interval 100 > logs/breakout/hyper/train_ballrel.txt

python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 500000 --num-steps 500 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 2000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 100 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 1 --log-interval 100 --interaction-probability 0 --max-steps 300 --pretest-trials 20 --temporal-extend 300 --interaction-prediction 0 --breakout-variant single_block --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --terminate-cutoff --test-episode --max-critic 100 --log-interval 25 --save-graph data/breakout/single_block1 --save-interval 100 > logs/breakout/variant_trials/single/train_single_block1.txt
python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 500000 --num-steps 500 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 2000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 100 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 1 --log-interval 100 --interaction-probability 0 --max-steps 300 --pretest-trials 20 --temporal-extend 300 --interaction-prediction 0 --breakout-variant single_block --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --terminate-cutoff --test-episode --max-critic 100 --log-interval 25 --save-graph data/breakout/single_block2 --save-interval 100 > logs/breakout/variant_trials/single/train_single_block2.txt
python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 500000 --num-steps 500 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 2000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 100 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 1 --log-interval 100 --interaction-probability 0 --max-steps 300 --pretest-trials 20 --temporal-extend 300 --interaction-prediction 0 --breakout-variant single_block --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --terminate-cutoff --test-episode --max-critic 100 --log-interval 25 --save-graph data/breakout/single_block3 --save-interval 100 > logs/breakout/variant_trials/single/train_single_block3.txt