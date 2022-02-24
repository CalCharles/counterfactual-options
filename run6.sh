# python train_option.py --dataset-dir data/robopushing/interaction_gbv/ --graph-dir data/robopushing/gripper_graph --object Block --env RoboPushing --option-type model --buffer-len 1000000 --num-steps 200 --gamma .99 --batch-size 128 --num-iters 25000 --terminal-type param --reward-type param --sampler-type cuuni --sample-schedule 50000 --sample-distance .15 --parameterized-lambda 1 --epsilon-close 0.005 --hidden-sizes 128 128 128 128 --learning-type hersac --grad-epoch 75 --pretrain-iters 20000 --lr 1e-4 --epsilon .1 --param-recycle .1 --sample-continuous 2 --select-positive .3 --gpu 2 --tau .001 --log-interval 500 --reward-constant -1 --time-cutoff 200 --resample-timer 0 --max-steps 50 --print-test --pretest-trials 20 --hardcode-norm robopush 3 1 --observation-setting 1 0 0 1 1 1 0 0 0 --interaction-probability 0 --interaction-prediction 0 --her-only-interact 2 --relative-action .1 --temporal-extend 4 --prioritized-replay 0.2 0.4 --early-stopping 1 --use-termination --use-pair-gamma --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/blocksd15/ --log-only --save-graph data/robopushing/block_graphsd15 --save-interval 100 > logs/robopushing/train_blocksd15.txt &

# python train_option.py --dataset-dir data/robopushing/interaction_gbv/ --graph-dir data/robopushing/gripper_graph --object Block --env RoboPushing --option-type model --buffer-len 1000000 --num-steps 200 --gamma .99 --batch-size 128 --num-iters 25000 --terminal-type param --reward-type param --sampler-type cuuni --sample-schedule 50000 --sample-distance .2 --parameterized-lambda 1 --epsilon-close 0.005 --hidden-sizes 128 128 128 128 128 128 --learning-type hersac --grad-epoch 75 --pretrain-iters 20000 --lr 1e-4 --epsilon .1 --param-recycle .1 --sample-continuous 2 --select-positive .3 --gpu 2 --tau .001 --log-interval 500 --reward-constant -1 --time-cutoff 200 --resample-timer 0 --max-steps 50 --print-test --pretest-trials 20 --hardcode-norm robopush 3 1 --observation-setting 1 0 0 1 1 1 0 0 0 --interaction-probability 0 --interaction-prediction 0 --her-only-interact 2 --relative-action .1 --temporal-extend 4 --prioritized-replay 0.2 0.4 --early-stopping 1 --use-termination --use-pair-gamma --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/blocklayer/ --log-only --save-graph data/robopushing/block_graphlayer --save-interval 100 > logs/robopushing/train_blocklayer.txt &

# python train_option.py --dataset-dir data/robopushing/interaction_gbv/ --graph-dir data/robopushing/gripper_graph --object Block --env RoboPushing --option-type model --buffer-len 1000000 --num-steps 200 --gamma .99 --batch-size 128 --num-iters 25000 --terminal-type param --reward-type param --sampler-type cuuni --sample-schedule 50000 --sample-distance .2 --parameterized-lambda 1 --epsilon-close 0.005 --hidden-sizes 256 256 256 256 --learning-type hersac --grad-epoch 75 --pretrain-iters 20000 --lr 1e-4 --epsilon .1 --param-recycle .1 --sample-continuous 2 --select-positive .3 --gpu 3 --tau .001 --log-interval 500 --reward-constant -1 --time-cutoff 200 --resample-timer 0 --max-steps 50 --print-test --pretest-trials 20 --hardcode-norm robopush 3 1 --observation-setting 1 0 0 1 1 1 0 0 0 --interaction-probability 0 --interaction-prediction 0 --her-only-interact 2 --relative-action .1 --temporal-extend 4 --prioritized-replay 0.2 0.4 --early-stopping 1 --use-termination --use-pair-gamma --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/blockwide/ --log-only --save-graph data/robopushing/block_graphwide --save-interval 100 > logs/robopushing/train_blockwide.txt &

# python train_option.py --dataset-dir data/robopushing/interaction_gbv/ --graph-dir data/robopushing/gripper_graph --object Block --env RoboPushing --option-type model --buffer-len 1000000 --num-steps 200 --gamma .99 --batch-size 128 --num-iters 25000 --terminal-type param --reward-type param --sampler-type cuuni --sample-schedule 50000 --sample-distance .1 --parameterized-lambda 1 --epsilon-close 0.005 --hidden-sizes 128 128 128 128 --learning-type hersac --grad-epoch 75 --pretrain-iters 20000 --lr 1e-4 --epsilon .1 --param-recycle .1 --sample-continuous 2 --select-positive .3 --gpu 3 --tau .001 --log-interval 500 --reward-constant -1 --time-cutoff 200 --resample-timer 0 --max-steps 50 --print-test --pretest-trials 20 --hardcode-norm robopush 3 1 --observation-setting 1 0 0 1 1 1 0 0 0 --interaction-probability 0 --interaction-prediction 0 --her-only-interact 2 --relative-action .1 --temporal-extend 4 --prioritized-replay 0.2 0.4 --early-stopping 1 --use-termination --use-pair-gamma --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/blocksd10/ --log-only --save-graph data/robopushing/block_graphsd10 --save-interval 100 > logs/robopushing/train_blocksd10.txt &

# python train_option.py --dataset-dir data/robopushing/interaction_gbv/ --graph-dir data/robopushing/gripper_graph --object Block --env RoboPushing --option-type model --buffer-len 1000000 --num-steps 200 --gamma .99 --batch-size 128 --num-iters 25000 --terminal-type param --reward-type param --sampler-type cuuni --sample-schedule 50000 --sample-distance .3 --parameterized-lambda 1 --epsilon-close 0.005 --hidden-sizes 128 128 128 128 --learning-type hersac --grad-epoch 75 --pretrain-iters 20000 --lr 1e-4 --epsilon .1 --param-recycle .1 --sample-continuous 2 --select-positive .3 --gpu 3 --tau .001 --log-interval 500 --reward-constant -1 --time-cutoff 200 --resample-timer 0 --max-steps 50 --print-test --pretest-trials 20 --hardcode-norm robopush 3 1 --observation-setting 1 0 0 1 1 1 0 0 0 --interaction-probability 0 --interaction-prediction 0 --her-only-interact 2 --relative-action .1 --temporal-extend 4 --prioritized-replay 0.2 0.4 --early-stopping 1 --use-termination --use-pair-gamma --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/blocksd30/ --log-only --save-graph data/robopushing/block_graphsd30 --save-interval 100 > logs/robopushing/train_blocksd30.txt

python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 50000 --num-steps 2000 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 3000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 200 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 2 --log-interval 100 --interaction-probability 0 --max-steps 3000 --pretest-trials 5 --test-trials 5 --temporal-extend 300 --interaction-prediction 0 --breakout-variant center_large --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --test-episode --max-critic 150 --log-interval 25 --save-graph data/breakout/center_large1 --save-interval 100 > logs/breakout/variant_trials/center/train_center_large1.txt
python train_option.py --dataset-dir dummy --graph-dir data/breakout/ball_graph --object Reward --option-type model --sampler-type randblock --policy-type pair --buffer-len 50000 --num-steps 2000 --gamma 0.99 --batch-size 128 --num-iters 1000 --terminal-type inter --reward-type tscale --reward-constant -1 --parameterized-lambda 1 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff 3000 --train --hidden-sizes 128 128 256 1024 --learning-type rainbow --grad-epoch 200 --pretrain-iters 10000 --lr 1e-5 --tau .001 --epsilon 0.1 --gpu 2 --log-interval 100 --interaction-probability 0 --max-steps 3000 --pretest-trials 5 --test-trials 5 --temporal-extend 300 --interaction-prediction 0 --breakout-variant center_large --observation-setting 1 0 0 0 0 0 0 0 0 --discretize-actions --env-reset --hardcode-norm breakout 4 1 --only-termination --use-interact --sum-rewards --test-episode --max-critic 150 --log-interval 25 --save-graph data/breakout/center_large2 --save-interval 100 > logs/breakout/variant_trials/center/train_center_large2.txt
