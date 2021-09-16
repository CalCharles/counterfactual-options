# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --posttrain-iters 50000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bpA > logs/ball_paddle_interactionA.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bpA --save-dir data/interaction_bpsA > logs/ball_paddle_int_testA.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --posttrain-iters 50000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp1 > logs/ball_paddle_interaction1.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bp1 --save-dir data/interaction_bps1 > logs/ball_paddle_int_test1.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --posttrain-iters 50000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp3 > logs/ball_paddle_interaction3.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bp2 --save-dir data/interaction_bps2 > logs/ball_paddle_int_test2.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --posttrain-iters 50000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp3 > logs/ball_paddle_interaction3.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bp3 --save-dir data/interaction_bps3 > logs/ball_paddle_int_test3.txt


# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 0 --epsilon-schedule 1000 --num-iters 0 --interaction-iters 0 --posttrain-iters 100000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp3p1 --dataset-dir data/interaction_bp3 --load-weights > logs/ball_paddle_interaction3p1.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bp3p1 --save-dir data/interaction_bps3p1 > logs/ball_paddle_int_test3p1.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 0 --epsilon-schedule 1000 --num-iters 0 --interaction-iters 0 --posttrain-iters 100000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp3p2 --dataset-dir data/interaction_bp3 --load-weights > logs/ball_paddle_interaction3p2.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bp3p2 --save-dir data/interaction_bps3p2 > logs/ball_paddle_int_test3p2.txt

# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 0 --epsilon-schedule 1000 --num-iters 0 --interaction-iters 0 --posttrain-iters 100000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bpAp1 --dataset-dir data/interaction_bpA --load-weights > logs/ball_paddle_interactionAp1.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bpAp1 --save-dir data/interaction_bpsAp1 > logs/ball_paddle_int_testAp1.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 0 --epsilon-schedule 1000 --num-iters 0 --interaction-iters 0 --posttrain-iters 100000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bpAp2 --dataset-dir data/interaction_bpA --load-weights > logs/ball_paddle_interactionAp2.txt
# python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bpAp2 --save-dir data/interaction_bpsAp2 > logs/ball_paddle_int_testAp2.txt

# base te
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballte6.txt
# # reward scaling, reward normalization
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.01 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballc01pl100.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.1 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballc1pl100.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 10 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballc0pl10.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 10 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.1 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballc1pl10.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant 0.0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.4 --temporal-extend 6 --reward-normalization > logs/ball_hyperparams/train_ballrn.txt
# # prioritized replay: alpha is rate of sample, beta is importance sampling weight
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.01 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.6 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballpr64.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.01 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.4 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballpr44.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.01 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.9 0.4 --temporal-extend 6 > logs/ball_hyperparams/train_ballpr94.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.01 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.5 --temporal-extend 6 > logs/ball_hyperparams/train_ballpr85.txt
# python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 50 --reward-constant -0.01 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --relative-state --relative-action 8 --pretest-trials 20 --prioritized-replay 0.8 0.3 --temporal-extend 6 > logs/ball_hyperparams/train_ballpr83.txt
# velocity schedule
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 3000 --terminal-type comb --reward-type tcomb --interaction-lambda 10 --parameterized-lambda 100 --true-reward-lambda 10 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 50 > logs/param_tests/train_ball_velocity10.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 3000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 1000 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 50 --param-interaction > logs/param_tests/train_ball_velocity100.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 3000 --terminal-type comb --reward-type tcomb --interaction-lambda 10 --parameterized-lambda 100 --true-reward-lambda 10 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 2 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 50 > logs/param_tests/train_ball_velocity10te2.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 3000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 1000 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 2 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 > logs/param_tests/train_ball_velocity100te2.txt

# # baseline method
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction --save-graph data/ball_vel_graphs --save-interval 100 > logs/param_tests/train_ball_velocitys.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocity.txt

# # reward constant
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant -.1 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocityrcp1.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant -1 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocityrc1.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant -2 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocityrc2.txt
# python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.03 --behavior-type greedyQ --select-positive .3 --gpu 2 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction > logs/param_tests/train_ball_velocityep03.txt

# reward changes
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 75 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 0 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bv1-100--1.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 75 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 0 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bv1-200--1.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 75 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 0 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bv100-200--1.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 10 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 75 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -.01 --interaction-probability 0 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bv1-10--01.txt
# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 2000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 0 --parameterized-lambda 100 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 75 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -.01 --interaction-probability 0 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bv0-100--01.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 3000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 256 256 256 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .003 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --save-graph data/breakout/hypertest/bvti100-200--1t1 --save-interval 100 > logs/breakout/hyper/bvti100-200--1t1.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 5000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 > logs/breakout/hyper/bvit1-200--1t1.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 5000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 2 1 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --record-rollouts data/breakout/ballt1 --save-graph data/breakout/ball_grapht1 --save-interval 100 > logs/breakout/ballt1.txt

# python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 1000000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 3000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 3 1 --her-only-interact 1 --resample-timer 0 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --env-reset --record-rollouts data/breakout/hyper/ball --save-graph data/breakout/hyper/ball_graph --save-interval 100 > logs/breakout/hyper/train_ball.txt

# using sample merged with defualt parameters
python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 1000000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 20000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .5 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 3 1 --her-only-interact 1 --resample-timer 0 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --env-reset --sample-merged --save-graph data/breakout/hyper/ball_graphm --save-interval 100 > logs/breakout/hyper/train_ballm.txt &

python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 1000000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 20000 --terminal-type tcomb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 200 --true-reward-lambda 0 --epsilon-close .5 --time-cutoff 300 --train --hidden-sizes 128 128 128 128 128 --learning-type herddpg --grad-epoch 50 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --select-positive .2 --tau .001 --log-interval 100 --reward-constant -1 --interaction-probability 1 --interaction-prediction 0 --max-steps 300 --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 20 --param-interaction --use-termination --gpu 0 --hardcode-norm breakout 3 1 --her-only-interact 1 --resample-timer 0 --drop-stopping --observation-setting 1 0 0 1 0 1 0 0 0 --env-reset --sample-merged --save-graph data/breakout/hyper/ball_graphsp2 --save-interval 100 > logs/breakout/hyper/train_ballsp2.txt
