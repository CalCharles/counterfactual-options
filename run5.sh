python train_option.py --dataset-dir dummy --env RoboPushing --num-obstacles 15 --graph-dir data/robopushing/block_graph --object Reward --option-type model --policy-type pair --buffer-len 500000 --num-steps 1000 --gamma 0.99 --batch-size 128 --num-iters 10000 --terminal-type param --reward-type negparam --parameterized-lambda 10 --true-reward-lambda -2 --reward-constant -0.2 --epsilon-close .05 --param-norm 2 --negative-epsilon-close .03 --time-cutoff 150 --train --hidden-sizes 128 256 512 1024 128 --learning-type herddpg --grad-epoch 300 --pretrain-iters 100000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type tar --select-positive .2 --gpu 1 --resample-timer -1 --tau .001 --log-interval 50 --max-steps 150 --print-test --pretest-trials 20 --hardcode-norm robopush 4 1 --interaction-probability 0 --interaction-prediction 0 --use-termination --observation-setting 1 0 0 1 1 1 0 0 0 --relative-action .15 --temporal-extend 20 --target Target --sum-reward --prioritized-replay 0.2 0.4 --param-contained --interleave --terminate-reset --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/reward_ra15 --log-only --save-graph /hdd/robopushing/live_record/graphs/reward_graph_ra15 --save-interval 100 > logs/robopushing/reward_tuning/train_reward_ra15.txt &

python train_option.py --dataset-dir dummy --env RoboPushing --num-obstacles 15 --graph-dir data/robopushing/block_graph --object Reward --option-type model --policy-type pair --buffer-len 500000 --num-steps 1000 --gamma 0.99 --batch-size 128 --num-iters 10000 --terminal-type param --reward-type negparam --parameterized-lambda 10 --true-reward-lambda -2 --reward-constant -0.2 --epsilon-close .05 --param-norm 2 --negative-epsilon-close .03 --time-cutoff 150 --train --hidden-sizes 128 256 512 1024 128 --learning-type herddpg --grad-epoch 300 --pretrain-iters 100000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type tar --select-positive .2 --gpu 1 --resample-timer -1 --tau .001 --log-interval 50 --max-steps 150 --print-test --pretest-trials 20 --hardcode-norm robopush 4 1 --interaction-probability 0 --interaction-prediction 0 --use-termination --observation-setting 1 0 0 1 1 1 0 0 0 --relative-action .1 --temporal-extend 20 --target Target --sum-reward --prioritized-replay 0.2 0.4 --param-contained --interleave --terminate-reset --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/reward_ra10 --log-only --save-graph /hdd/robopushing/live_record/graphs/reward_graph_ra10 --save-interval 100 > logs/robopushing/reward_tuning/train_reward_ra10.txt 

python train_option.py --dataset-dir dummy --env RoboPushing --num-obstacles 15 --graph-dir data/robopushing/block_graph --object Reward --option-type model --policy-type pair --buffer-len 500000 --num-steps 1000 --gamma 0.99 --batch-size 128 --num-iters 10000 --terminal-type param --reward-type negparam --parameterized-lambda 10 --true-reward-lambda -2 --reward-constant -0.2 --epsilon-close .05 --param-norm 2 --negative-epsilon-close .03 --time-cutoff 150 --train --hidden-sizes 128 256 512 1024 128 --learning-type herddpg --grad-epoch 300 --pretrain-iters 100000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type tar --select-positive .2 --gpu 2 --resample-timer -1 --tau .001 --log-interval 50 --max-steps 150 --print-test --pretest-trials 20 --hardcode-norm robopush 4 1 --interaction-probability 0 --interaction-prediction 0 --use-termination --observation-setting 1 0 0 1 1 1 0 0 0 --relative-action .05 --temporal-extend 20 --target Target --sum-reward --prioritized-replay 0.2 0.4 --param-contained --interleave --terminate-reset --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/reward_ra05 --log-only --save-graph /hdd/robopushing/live_record/graphs/reward_graph_ra05 --save-interval 100 > logs/robopushing/reward_tuning/train_reward_ra05.txt &

python train_option.py --dataset-dir dummy --env RoboPushing --num-obstacles 15 --graph-dir data/robopushing/block_graph --object Reward --option-type model --policy-type pair --buffer-len 500000 --num-steps 1000 --gamma 0.9 --batch-size 128 --num-iters 10000 --terminal-type param --reward-type negparam --parameterized-lambda 10 --true-reward-lambda -2 --reward-constant -0.2 --epsilon-close .05 --param-norm 2 --negative-epsilon-close .03 --time-cutoff 150 --train --hidden-sizes 128 256 512 1024 128 --learning-type herddpg --grad-epoch 300 --pretrain-iters 100000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type tar --select-positive .2 --gpu 2 --resample-timer -1 --tau .001 --log-interval 50 --max-steps 150 --print-test --pretest-trials 20 --hardcode-norm robopush 4 1 --interaction-probability 0 --interaction-prediction 0 --use-termination --observation-setting 1 0 0 1 1 1 0 0 0 --relative-action .1 --temporal-extend 20 --target Target --sum-reward --prioritized-replay 0.2 0.4 --param-contained --interleave --terminate-reset --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/rewardg90 --log-only --save-graph /hdd/robopushing/live_record/graphs/reward_graphg90 --save-interval 100 > logs/robopushing/reward_tuning/train_rewardg90.txt &

python train_option.py --dataset-dir dummy --env RoboPushing --num-obstacles 15 --graph-dir data/robopushing/block_graph --object Reward --option-type model --policy-type pair --buffer-len 500000 --num-steps 1000 --gamma 0.99 --batch-size 128 --num-iters 10000 --terminal-type param --reward-type negparam --parameterized-lambda 10 --true-reward-lambda -2 --reward-constant -1 --epsilon-close .05 --param-norm 2 --negative-epsilon-close .03 --time-cutoff 150 --train --hidden-sizes 128 256 512 1024 128 --learning-type herddpg --grad-epoch 300 --pretrain-iters 100000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type tar --select-positive .2 --gpu 3 --resample-timer -1 --tau .001 --log-interval 50 --max-steps 150 --print-test --pretest-trials 20 --hardcode-norm robopush 4 1 --interaction-probability 0 --interaction-prediction 0 --use-termination --observation-setting 1 0 0 1 1 1 0 0 0 --relative-action .1 --temporal-extend 20 --target Target --sum-reward --prioritized-replay 0.2 0.4 --param-contained --interleave --terminate-reset --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/rewardrc10 --log-only --save-graph /hdd/robopushing/live_record/graphs/reward_graphrc10 --save-interval 100 > logs/robopushing/reward_tuning/train_rewardrc10.txt 

python train_option.py --dataset-dir dummy --env RoboPushing --num-obstacles 15 --graph-dir data/robopushing/block_graph --object Reward --option-type model --policy-type pair --buffer-len 500000 --num-steps 1000 --gamma 0.99 --batch-size 128 --num-iters 10000 --terminal-type param --reward-type negparam --parameterized-lambda 10 --true-reward-lambda -2 --reward-constant -0.2 --epsilon-close .05 --param-norm 2 --negative-epsilon-close .03 --time-cutoff 150 --train --hidden-sizes 128 256 512 1024 128 --learning-type hersac --grad-epoch 300 --pretrain-iters 100000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type tar --select-positive .2 --gpu 3 --resample-timer -1 --tau .001 --log-interval 50 --max-steps 150 --print-test --pretest-trials 20 --hardcode-norm robopush 4 1 --interaction-probability 0 --interaction-prediction 0 --use-termination --observation-setting 1 0 0 1 1 1 0 0 0 --relative-action .1 --temporal-extend 20 --target Target --sum-reward --prioritized-replay 0.2 0.4 --param-contained --interleave --terminate-reset --record-rollouts /hdd/datasets/counterfactual_data/robopushing/live_record/rewardsac --log-only --save-graph /hdd/robopushing/live_record/graphs/reward_graphsac --save-interval 100 > logs/robopushing/reward_tuning/train_rewardsac.txt
