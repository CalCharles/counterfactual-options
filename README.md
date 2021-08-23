requirements: pytorch, opencv, imageio

python generate_breakout_data.py data/random 3000

python construct_dataset_model.py --graph-dir data/graph/ --dataset-dir data/random/ --num-frames 1000 --object Action
python add_option.py --dataset-dir data/random/ --object Paddle --buffer-steps 5 --num-steps 5 --factor 8 --learning-type a2c --gamma .1 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --record-rollouts data/paddle --num-iters 10000 --save-graph data/tempgraph --train --save-interval 1000 > paddletrain.txt

python construct_dataset_model.py --graph-dir data/paddlegraph/ --dataset-dir data/paddle/ --num-frames 10000 --object Paddle --target Ball

python construct_hypothesis_model.py --dataset-dir data/random/ --log-interval 500 --factor 8 --num-layers 2 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 0 --epsilon-schedule 3000 --num-iters 18000 --interaction-binary -1 -8 --save-dir data/interaction_ln > train_hypothesis.txt

<!-- python add_option.py --dataset-dir data/paddle/ --object Ball --buffer-steps 5 --num-steps 5 --factor 32 --learning-type a2c --gamma .99 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --num-iters 100000 --save-interval 1000 --save-graph data/tempgraph --set-time-cutoff --graph-dir data/paddlegraph --record-rollouts data/Ball

python add_option.py --dataset-dir data/paddle/ --object Ball --buffer-steps 5 --num-steps 5 --factor 32 --learning-type a2c --gamma .99 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --num-iters 1000000 --save-interval 1000 --save-graph data/tempgraph --set-time-cutoff --graph-dir data/paddlegraph --record-rollouts data/Ball --use-both 1 --min-use 0

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 5 --num-steps 5 --factor 16 --num-layers 2 --learning-type a2c --gamma .99 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --num-iters 1000000 --save-interval 1000 --save-graph data/tempgraph --set-time-cutoff --graph-dir data/paddlegraph --record-rollouts data/Ball --use-both 1 --min-use 5 > ball.txt

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 5 --num-steps 5 --factor 16 --num-layers 2 --learning-type a2c --gamma .99 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --num-iters 1000000 --set-time-cutoff --graph-dir data/paddlegraph --record-rollouts data/Ball --use-both 1 --min-use 5 

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 32 --num-steps 32 --factor 16 --num-layers 2 --learning-type a2c --gamma .99 --batch-size 32 --gpu 2 --normalize --entropy-coef .01 --num-iters 1000000 --set-time-cutoff --graph-dir data/paddlegraph --use-both 0 --min-use 0 --train --record-rollouts ../datasets/caleb_data/balla2c/ --save-interval 1000 --save-graph data/tempa2cgraph > balla2c.txt

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 10000 --num-steps 32 --factor 64 --num-layers 1 --learning-type dqn --gamma .99 --batch-size 16 --grad-epoch 5 --gpu 2 --normalize --entropy-coef .01 --warm-up 100 --num-iters 1000000 --behavior-type greedyQ --epsilon .9 --epsilon-schedule 10000 --set-time-cutoff --lr 1e-6 --graph-dir data/paddlegraph --use-both 0 --min-use 0 --Q-critic --train --record-rollouts ../datasets/caleb_data/balldqn/  --save-interval 1000 --save-graph data/tempdqngraph > balldqn.txt 

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 10000 --num-steps 32 --factor 64 --num-layers 1 --learning-type her --gamma .99 --batch-size 16 --grad-epoch 10 --gpu 3 --normalize --entropy-coef .01 --warm-up 100 --num-iters 1000000 --behavior-type greedyQ --Q-critic --epsilon .9 --epsilon-schedule 10000 --set-time-cutoff --lr 1e-6 --graph-dir data/paddlegraph --use-both 0 --min-use 0 --train --record-rollouts ../datasets/caleb_data/ballher/ --save-interval 1000 --save-graph data/temphergraph > ballher.txt 

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 10000 --num-steps 32 --factor 64 --num-layers 1 --learning-type gsr --gamma .99 --batch-size 16 --grad-epoch 10 --gpu 3 --normalize --entropy-coef .01 --warm-up 100 --num-iters 1000000 --behavior-type greedyQ --Q-critic --epsilon .95 --epsilon-schedule 10000 --set-time-cutoff --lr 1e-6 --graph-dir data/paddlegraph --use-both 0 --min-use 0 --train --record-rollouts ../datasets/caleb_data/ballgsr/ --save-interval 1000 --save-graph data/tempgsrgraph > ballgsr.txt 

python add_option.py --dataset-dir data/hackedpaddle/ --object Ball --buffer-steps 10000 --num-steps 32 --factor 64 --num-layers 1 --learning-type her --gamma .99 --batch-size 16 --grad-epoch 10 --gpu 3 --normalize --entropy-coef .01 --warm-up 100 --num-iters 1000000 --behavior-type greedyQ --Q-critic --epsilon .95 --epsilon-schedule 10000 --set-time-cutoff --lr 1e-6 --graph-dir data/paddlegraph --use-both 0 --min-use 0 --train --record-rollouts ../datasets/caleb_data/ballpri/ --save-interval 1000 --save-graph data/tempprigraph --prioritized-replay max_reward --weighting-lambda .05 > ballpri.txt 

python add_option.py --dataset-dir data/random/ --object Paddle --buffer-steps 100000 --num-steps 5 --factor 16 --warm-up 10000 --esilon 1 --epsilon-schedule 10000 --learning-type a2c --gamma .99 --batch-size 4 --grad-epoch 4 --gpu 2 --lr .00025 --option-type raw --policy-type image --normalize --num-iters 500000 --entropy-coef 0.01 --init-form orth > rawtestdqn.txt


Hacked commands
python construct_interaction_model.py --graph-dir data/PAgraph/ --dataset-dir data/random/ --num-frames 1000 --object Action --target Paddle > interactiontest.txt
python add_option.py --dataset-dir data/random/ --object Paddle --option-type hacked --buffer-steps 5 --num-steps 5 --gamma .1 --batch-size 5 --record-rollouts data/paddle --num-iters 1000 > hackedpaddletrain.txt
 -->

New interaction model
Train paddle model
python construct_hypothesis_model.py --record-rollouts data/random/ --log-interval 500 --factor 8 --num-layers 2 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 0 --epsilon-schedule 3000 --num-iters 100000 --interaction-binary -1 -8 10 --train --save-dir data/interaction_ln > train_hypothesis.txt

python add_option.py --dataset-dir data/interaction_ln/ --object Paddle --option-type model --buffer-steps 500000 --num-steps 50 --gamma .99 --batch-size 16 --record-rollouts data/paddle --num-iters 4000 --terminal-type comb --reward-type comb --parameterized-lambda 0 --epsilon-close 1 --time-cutoff 50 --set-time-cutoff --init-form none --train --normalize  --num-layers 3 --factor 8 --learning-type her --grad-epoch 50 --warm-up 100 --warm-update 100 --lr 1e-4 --epsilon .1 --epsilon-schedule 100 --behavior-type greedyQ --return-form none --Q-critic --select-positive .5 --gpu 1 --resample-timer 50 --double-Q 50 --log-interval 25 --reward-constant -1 --save-interval 100 --save-graph data/paddle_graph3 > logs/train_paddle_3.txt

python test_option.py --dataset-dir data/interaction_ln/ --object Paddle --option-type model --buffer-steps 1000 --gamma .99 --batch-size 5 --record-rollouts data/paddle_test --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 0 --epsilon-close 1 --normalize --lr 1e-5 --behavior-type greedyQ --gpu 1 --graph-dir data/paddle_graph/

Train ball model
python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 16 --num-layers 2 --predict-dynamics --action-shift --batch-size 32 --pretrain-iters 0 --epsilon-schedule 30000 --num-iters 1000000 --interaction-binary -1 -8 -10 --num-frames 1000000 --graph-dir data/paddle_graph/ --train --save-dir data/interaction_bp > train_ball_hypothesis.txt

python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -6 -10 --num-frames 50000 --graph-dir data/paddle_graph2/

python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 24 --num-layers 3 --batch-size 32 --pretrain-iters 0 --epsilon-schedule 1000 --num-iters 100000 --log-interval 1000 --interaction-binary -2 -6 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp > logs/trace_logs/train_ball_hypothesis.txt

python construct_hypothesis_model.py --record-rollouts data/paddle/ --log-interval 500 --factor 8 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 100000 --interaction-iters 0 --log-interval 1000 --interaction-binary -2 -6 -10 --num-frames 50000 --graph-dir data/paddle_graph2/ --dataset-dir data/interaction_bp --save-dir data/interaction_bp_vals> logs/ball_paddle_int_test.txt

python construct_hypothesis_model.py --record-rollouts data/paddle/ --factor 12 --num-layers 3 --batch-size 32 --pretrain-iters 100000 --epsilon-schedule 1000 --num-iters 200000 --posttrain-iters 100000 --log-interval 1000 --interaction-binary -2 -5 -10 --num-frames 100000 --graph-dir data/paddle_graph2/ --train --save-dir data/interaction_bp0 > logs/ball_paddle_interaction0.txt


python add_option.py --dataset-dir data/interaction_bps3p2/ --object Ball --option-type model --buffer-steps 500000 --num-steps 100 --gamma .99 --batch-size 16 --record-rollouts data/ball --num-iters 50000 --terminal-type comb --reward-type comb --continuous --parameterized-lambda 10 --epsilon-close .5 --time-cutoff 100 --set-time-cutoff --init-form none --train --normalize  --num-layers 3 --factor 16 --learning-type her --grad-epoch 50 --warm-up 100 --warm-update 100 --lr 1e-4 --epsilon .1 --epsilon-schedule 100 --behavior-type greedyQ --return-form none --Q-critic --select-positive .5 --gpu 1 --resample-timer 100 --double-Q 100 --log-interval 50 --reward-constant -.1 --graph-dir data/paddle_graph2 --save-graph data/ball_graph > logs/train_ball.txt



Nav2D:
python add_option.py --object Raw --option-type raw --true-environment --env Nav2D --buffer-steps 500000 --num-steps 50 --gamma .99 --batch-size 16 --record-rollouts data/paddle --num-iters 4000 --terminal-type true --reward-type true --epsilon-close 1 --time-cutoff 50 --set-time-cutoff --init-form none --train --normalize --policy-type grid --learning-type her --grad-epoch 50 --warm-up 400 --warm-update 0 --lr 1e-4 --epsilon .1 --behavior-type greedyQ --return-form none --Q-critic --select-positive .5 --gpu 2 --epsilon-schedule 250 --log-interval 25 --optim Adam --resample-timer 50 --factor 16 --double-Q 60

Pendulum
python add_option.py --object Raw --option-type raw --true-environment --env Pend-Gym --continuous --buffer-steps 500000 --num-steps 50 --gamma .99 --batch-size 16 --record-rollouts data/paddle --num-iters 4000 --terminal-type true --reward-type true --epsilon-close 1 --time-cutoff 50 --set-time-cutoff --init-form none --train --normalize --policy-type grid --learning-type her --grad-epoch 50 --warm-up 400 --warm-update 0 --lr 1e-4 --epsilon .1 --behavior-type greedyQ --return-form none --Q-critic --select-positive .5 --gpu 2 --epsilon-schedule 250 --log-interval 25 --optim Adam --resample-timer 50 --factor 16 --double-Q 60


<!-- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
 -->
conda create -n ts python=3.8
pip install tianshou
conda install imageio
pip install opencv-python

python add_option.py --object Raw --option-type raw --true-environment --env Pend-Gym --buffer-steps 500000 --num-steps 1 --gamma .99 --batch-size 128 --num-iters 20000 --terminal-type true --reward-type true --epsilon-close 1 --init-form none --train --normalize --policy-type actorcritic --learning-type ddpg --grad-epoch 5 --warm-up 128 --warm-update 0 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --return-form none --Q-critic --gpu 2 --log-interval 200 --optim Adam --factor 8 --num-layers 2 --use-layer-norm --double-Q .001 --actor-critic-optimizer

Baseline tests:
point transformer largent koinenet


python add_option.py --object Raw --option-type raw --true-environment --env Nav2D --buffer-steps 500000 --num-steps 50 --gamma .99 --batch-size 16 --num-iters 4000 --terminal-type true --reward-type true --epsilon-close 1 --time-cutoff 50 --policy-type grid --learning-type herdqn --grad-epoch 50 --warm-up 10000 --lr 1e-4 --epsilon .1 --return-form none --select-positive .5 --gpu 2 --epsilon-schedule 250 --log-interval 25 --resample-timer 50 --factor 16--tau 60


Gym tests:
# sac
python train_option.py --gpu 1 --hidden-sizes 128 128 128 --env gymenvPendulum-v0 --learning-type sac --actor-lr 1e-4 --critic-lr 1e-3 --tau .005 --num-steps 10 --true-environment --option-type raw --pretrain-iters 1000 --num-iters 7000 --buffer-len 100000

# ddpg
python train_option.py --gpu 1 --hidden-sizes 128 128 128 --env gymenvPendulum-v0 --learning-type ddpg --actor-lr 1e-4 --critic-lr 1e-3 --tau .005 --num-steps 10 --true-environment --option-type raw --pretrain-iters 1000 --num-iters 7000 --buffer-len 100000

# dqn
python train_option.py --gpu 1 --hidden-sizes 128 128 128 --env gymenvCartPole-v0 --learning-type dqn --actor-lr 1e-4 --critic-lr 1e-3 --tau 100 --num-steps 10 --true-environment --option-type raw --pretrain-iters 1000 --num-iters 2000 --buffer-len 50000

# ppo
python train_option.py --gpu 1 --hidden-sizes 64 64 --env gymenvCartPole-v0 --learning-type ppo --actor-lr 1e-3 --critic-lr 1e-3 --tau 100 --num-steps 64 --true-environment --option-type raw --pretrain-iters 1000 --num-iters 20000 --buffer-len 20000 --grad-epoch 48 --log-interval 25

python train_option.py --gpu 1 --hidden-sizes 64 64 --env gymenvCartPole-v0 --learning-type ppo --actor-lr 1e-3 --num-steps 64 --true-environment --option-type raw --pretrain-iters 1000 --num-iters 20000 --buffer-len 20000 --grad-epoch 48 --log-interval 25

python train_option.py --gpu 1 --hidden-sizes 128 128 128 --env gymenvPendulum-v0 --learning-type ppo --actor-lr 1e-3 --num-steps 200 --true-environment --option-type raw --buffer-len 20000 --num-iters 2000 --log-interval 10 --grad-epoch 125


Nav2D grid test
# DQN with HER
python train_option.py --object Raw --option-type raw --true-environment --env Nav2D --buffer-len 1000000 --num-steps 50 --gamma .99 --batch-size 16 --num-iters 1500 --terminal-type env --reward-type env  --epsilon-close 1 --time-cutoff 50 --policy-type grid --learning-type herdqn --grad-epoch 50 --pretrain-iters 20000 --lr 1e-4 --epsilon .1 --select-positive .5 --gpu 2 --epsilon-schedule 2000 --log-interval 1 --max-steps 50 --hidden-sizes 32 64 64 564 --tau 3000 --resample-timer 50 --test-trials 3


Running Tianshou
# generate random data
python generate_breakout_data.py data/random 3000

# train the paddle motion model
python construct_hypothesis_model.py --record-rollouts data/breakout/random/ --env SelfBreakout --log-interval 500 --hidden-sizes 1024 1024 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 10000 --epsilon-schedule 3000 --num-iters 20000 --interaction-binary -1 -8 -16 --train --interaction-prediction .3 --save-dir data/breakout/interaction_ap > logs/breakout/action_paddle_interaction.txt

# Run "test" to fill in parameter values for the hypothesis model
python construct_hypothesis_model.py --record-rollouts data/breakout/random/ --graph-dir data/breakout/action_graph/ --dataset-dir data/breakout/interaction_ap --env SelfBreakout --batch-size 64 --predict-dynamics --epsilon-schedule 1000 --action-shift --log-interval 1000 --num-frames 100000 --interaction-binary -1 -8 -16 --gpu 2 --save-dir data/breakout/interaction_apv > logs/breakout/action_paddle_interactionv.txt

# train the paddle policy with HER and DQN
python train_option.py --dataset-dir data/breakout/interaction_apv/ --object Paddle --option-type model --buffer-len 500000 --num-steps 75 --gamma .99 --batch-size 16 --num-iters 4000 --terminal-type comb --reward-type comb --parameterized-lambda 30 --epsilon-close 1 --time-cutoff 75 --set-time-cutoff  --hidden-sizes 128 128 --learning-type herdqn --grad-epoch 75 --pretrain-iters 1000 --lr 1e-4 --epsilon .1 --epsilon-schedule 200 --select-positive .5 --gpu 1 --resample-timer 20 --tau 2500 --log-interval 25 --reward-constant -2 --max-steps 30 --print-test --use-termination --param-recycle .1 --sample-continuous 2 --pretest-trials 20 --hardcode-norm breakout 1 5 --interaction-probability 0 --interaction-prediction 0 --record-rollouts data/breakout/paddle --save-graph data/breakout/paddle_graph --save-interval 100 > logs/breakout/train_paddle.txt

# test the paddle policy
python test_option.py --dataset-dir data/interaction_apv/ --graph-dir data/paddle_graph --object Paddle --buffer-len 500000 --gamma .99 --test-trials 20 --num-iters 10 --terminal-type comb --reward-type comb --parameterized-lambda 0 --epsilon-close 1 --time-cutoff 50 --set-time-cutoff --gpu 1 --reward-constant -1 --max-steps 30 --print-test --sample-continuous 2 --visualize-param nosave --record-rollouts data/paddle_test > logs/test_paddle.txt

# train the ball bounce interaction model
# note: alter feature_explorer.py ATM to speed up paddle-ball interaction
python construct_hypothesis_model.py --env SelfBreakout --record-rollouts data/breakout/paddle/ --graph-dir data/breakout/paddle_graph/ --hidden-sizes 512 512 --batch-size 64 --pretrain-iters 100000 --interaction-iters 100000 --epsilon-schedule 1000 --num-iters 200000 --posttrain-iters 0 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --train --gpu 2  --interaction-prediction .1 --train-pair Paddle Ball --save-dir data/breakout/interaction_pbI > logs/breakout/paddle_ball_interactionI.txt

# Run "test" to fill in parameter values for the hypothesis model
python construct_hypothesis_model.py --env SelfBreakout --record-rollouts data/breakout/paddle/ --graph-dir data/breakout/ball_graph/ --dataset-dir data/breakout/interaction_pbI --hidden-sizes 512 512 --batch-size 64 --epsilon-schedule 1000 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --gpu 2 --save-dir data/breakout/interaction_pbIvd > logs/breakout/paddle_ball_interactionIvd.txt

<!-- # train ball bouncing with SAC and negative true rewards
CUDA_VISIBLE_DEVICES=1 python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --num-iters 1000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.6 0.4 --temporal-extend 6 --sac-alpha .5 --lookahead 4 --print-test --use-termination --record-rollouts data/ball --save-graph data/ball_bounce_graph2 --save-interval 100 > logs/train_ball_bounce2.txt

# no termination on bounce
python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff -1 --train  --hidden-sizes 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.6 0.4 --temporal-extend 6 --sac-alpha .5 --lookahead 4 --print-test --save-graph data/ball_bounce_graphnt --save-interval 100 > logs/train_ball_bouncent.txt

python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 64 --record-rollouts data/ball --num-iters 1000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 100 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 150 --train  --hidden-sizes 256 256 --learning-type sac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon .05 --behavior-type greedyQ --select-positive .5 --gpu 1 --resample-timer 150 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 100 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.6 0.4 --temporal-extend 6 --sac-alpha .5 --lookahead 4 --print-test --use-termination --save-graph data/ball_bounce_graph2 --save-interval 100 > logs/param_tests/train_ball_bounce.txt

# train with retargeted actions and PPO:
python train_option.py --dataset-dir data/interaction_bpIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 128 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type comb --parameterized-lambda 10 --epsilon-close .5 --time-cutoff 150 --train --hidden-sizes 512 512 --learning-type ppo --grad-epoch 64 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --select-positive .5 --gpu 1 --resample-timer 150 --tau .005 --log-interval 25 --reward-constant -.1 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 50 --pretest-trials 20 --relative-state --relative-action 6 --discretize-actions
 -->
# train ball bouncing with SAC-HER velocity
# 100, 200 reward
python train_option.py --dataset-dir data/breakout/interaction_pbIv/ --graph-dir data/breakout/paddle_graph --object Ball --option-type model --buffer-len 500000 --num-steps 250 --gamma .99 --batch-size 128 --num-iters 5000 --terminal-type comb --reward-type tcomb --interaction-lambda 100 --parameterized-lambda 200 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 100 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 150 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --resample-timer 100 --tau .001 --log-interval 50 --reward-constant 0 --interaction-probability 0 --max-steps 75  --relative-action 6 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 10 --param-interaction --use-termination --gpu 1 --observation-setting 1 0 0 1 0 1 0 0 0 --record-rollouts data/breakout/ball --save-graph data/breakout/ball_vel_graphs --save-interval 100 > logs/breakout/train_ball_velocity.txt

<!-- # no done termination
python train_option.py --dataset-dir data/interaction_pbIv/ --object Ball --option-type model --buffer-len 500000 --num-steps 300 --gamma .99 --batch-size 128 --record-rollouts data/ball --num-iters 2000 --terminal-type comb --reward-type tcomb --interaction-lambda 1 --parameterized-lambda 2 --true-reward-lambda 1 --epsilon-close .5 --time-cutoff -1 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 200 --pretrain-iters 10000 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --select-positive .3 --gpu 1 --resample-timer 500 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/paddle_graph --interaction-probability 1 --max-steps 75 --relative-state --relative-action 4 --pretest-trials 20 --prioritized-replay 0.2 0.4 --temporal-extend 6 --sac-alpha .3 --lookahead 4 --print-test --force-mask 0 0 1 1 0 --use-interact --max-hindsight 40 --param-interaction  --save-graph data/ball_vel_graphnt --save-interval 100 > logs/param_tests/train_ball_velocitynt.txt
 -->

# test SAC-HER velocity model
<!-- python test_option.py --dataset-dir data/interaction_pbIv/ --graph-dir data/ball_vel_graphs --object Ball --buffer-len 500000 --gamma .99 --record-rollouts data/ball_test --test-trials 500 --num-iters 5 --terminal-type comb --reward-type tcomb --option-type forward --interaction-lambda 100 --parameterized-lambda 1000 --true-reward-lambda 100 --epsilon-close 1 --time-cutoff -1 --set-time-cutoff --relative-state --relative-action 4 --gpu 3 --reward-constant 0 --max-steps 5000 --print-test --interaction-probability 1 --force-mask 0 0 1 1 0 --use-interact --use-termination --max-hindsight 50 --param-interaction --change-option > logs/test_vel.txt
 -->
python test_option.py --dataset-dir data/interaction_pbIv/ --graph-dir data/ball_vel_graphs --object Ball --buffer-len 500000 --gamma .99  --test-trials 100 --num-iters 5 --terminal-type comb --reward-type tcomb --option-type forward --interaction-lambda 100 --parameterized-lambda 1000 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff -1 --set-time-cutoff --relative-state --relative-action 4 --reward-constant 0 --max-steps 5000 --print-test --interaction-probability 1 --force-mask 0 0 1 1 0 --use-interact --use-termination --max-hindsight 50 --param-interaction --change-option --gpu 3 --record-rollouts data/ball_test --save-dir data/vel_model_option --save-graph data/vel_model_graph > logs/test_vel.txt

<!-- # train block interaction model -->
<!-- python construct_hypothesis_model.py --env SelfBreakout --record-rollouts data/ball_test/ --hidden-sizes 512 512 --batch-size 64 --pretrain-iters 100000 --interaction-iters 100000 --epsilon-schedule 1000 --num-iters 200000 --posttrain-iters 0 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --graph-dir data/ball_vel_graphs/ --train --gpu 2  --interaction-prediction .3 --policy-type pair --interaction-weight 10 --save-dir data/interaction_bbI > logs/ball_block_interactionI.txt
 -->
<!-- python construct_hypothesis_model.py --env SelfBreakout --record-rollouts data/ball_test/ --hidden-sizes 512 512 --batch-size 64 --pretrain-iters 20000 --interaction-iters 30000 --epsilon-schedule 1000 --num-iters 30000 --posttrain-iters 0 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --graph-dir data/ball_vel_graphs/ --train  --interaction-prediction .3 --policy-type pair --interaction-weight 10 --predict-dynamics --gpu 2 --save-dir data/interaction_bbI > logs/ball_block_interactionI.txt
 -->
# train block interaction model
python construct_hypothesis_model.py --record-rollouts data/ball_test/ --graph-dir data/ball_vel_graphs/ --env SelfBreakout --hidden-sizes 512 512 --batch-size 64 --pretrain-iters 10000 --interaction-iters 20000 --epsilon-schedule 1000 --num-iters 40000 --posttrain-iters 0 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --train --gpu 2  --interaction-prediction .3 --policy-type pair --interaction-weight 10 --predict-dynamics --multi-instanced --save-dir data/interaction_bbI > logs/ball_block_interactionI.txt

# test block interaction model
python construct_hypothesis_model.py --record-rollouts data/ball_test/ --graph-dir data/ball_vel_graphs/ --env SelfBreakout --hidden-sizes 512 512 --batch-size 64 --epsilon-schedule 1000 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --gpu 2 --dataset-dir data/interaction_bbI --save-dir data/interaction_bbIv > logs/ball_block_interactionIv.txt

<!-- python construct_hypothesis_model.py --env SelfBreakout --record-rollouts data/ball_test/ --hidden-sizes 512 512 --batch-size 64 --epsilon-schedule 1000 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --graph-dir data/ball_vel_graphs/ --gpu 2 --dataset-dir data/interaction_ballblock -->

# train block targeting policy
python train_option.py --dataset-dir data/interaction_bbIv/ --graph-dir data/vel_model_graph --object Block --option-type model --buffer-len 500000 --num-steps 500 --gamma .75 --batch-size 32 --num-iters 10000 --terminal-type proxist --reward-type proxist --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 500 --train --hidden-sizes 256 256 256 --learning-type herdqn --grad-epoch 250 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type inst --select-positive .5 --gpu 2 --resample-timer 1 --tau .001 --log-interval 50 --reward-constant -1 --interaction-probability 0 --max-steps 150 --pretest-trials 20 --temporal-extend 300 --sac-alpha .2 --lookahead 4 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --max-distance-epsilon 6 --record-rollouts data/block --save-graph data/block_graph_model --save-interval 100 > logs/train_block_model.txt

python train_option.py --dataset-dir data/interaction_bbIv/ --graph-dir data/vel_model_graph --object Block --option-type model --buffer-len 500000 --num-steps 500 --gamma .75 --batch-size 32 --num-iters 10000 --terminal-type proxist --reward-type proxist --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 500 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 250 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type inst --select-positive .5 --gpu 2 --resample-timer 1 --tau .001 --log-interval 50 --reward-constant -1 --interaction-probability 0 --max-steps 200 --pretest-trials 20 --temporal-extend 300 --sac-alpha .2 --lookahead 4 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --max-distance-epsilon 5 --record-rollouts data/block_sac --save-graph data/block_graph_model_sac --save-interval 100 > logs/train_block_model_sac.txt

<!-- python train_option.py --dataset-dir data/interaction_bbIv/ --object Block --option-type model --buffer-len 500000 --num-steps 500 --gamma .99 --batch-size 32 --record-rollouts data/block --num-iters 2000 --terminal-type proxist --reward-type proxist --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 10 --epsilon-close .5 --time-cutoff 1000 --train  --hidden-sizes 256 256 256 --learning-type herdqn --grad-epoch 250 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type inst --select-positive .5 --gpu 1 --resample-timer 1 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/ball_vel_graphs --interaction-probability 0 --max-steps 200 --pretest-trials 20 --temporal-extend 300 --sac-alpha .2 --lookahead 4 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --max-distance-epsilon 6 --save-graph data/block_graph --save-interval 100 > logs/train_block.txt

python train_option.py --dataset-dir data/interaction_bbIv/ --object Block --option-type model --buffer-len 500000 --num-steps 500 --gamma .99 --batch-size 32 --record-rollouts data/block --num-iters 2000 --terminal-type proxist --reward-type proxist --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 10 --epsilon-close .5 --time-cutoff 1000 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 250 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type inst --select-positive .5 --gpu 1 --resample-timer 1 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/ball_vel_graphs --interaction-probability 0 --max-steps 200 --pretest-trials 2 --temporal-extend 300 --sac-alpha .2 --lookahead 4 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --max-distance-epsilon 6 --test-trials

python train_option.py --dataset-dir data/interaction_bbIv/ --object Block --option-type model --buffer-len 500000 --num-steps 500 --gamma .99 --batch-size 32 --record-rollouts data/block --num-iters 2000 --terminal-type proxist --reward-type proxist --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 10 --epsilon-close .5 --time-cutoff 1000 --train  --hidden-sizes 256 256 256 --learning-type hersac --grad-epoch 250 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type inst --select-positive .5 --gpu 1 --resample-timer 1 --tau .001 --log-interval 50 --reward-constant 0 --graph-dir data/ball_vel_graphs --interaction-probability 0 --max-steps 200 --pretest-trials 20 --temporal-extend 300 --sac-alpha .2 --lookahead 4 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --max-distance-epsilon 6 > out.txt
 -->

<!--  python train_option.py --dataset-dir data/newrun/interaction_bbIv/ --graph-dir data/newrun/vel_model_graph --object Block --option-type model --buffer-len 500000 --num-steps 500 --gamma .75 --batch-size 32 --num-iters 10000 --terminal-type proxist --reward-type proxist --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 1000 --train  --hidden-sizes 256 256 256 --learning-type herdqn --grad-epoch 250 --pretrain-iters 10000 --lr 1e-4 --epsilon 0.1 --behavior-type greedyQ --sampler-type inst --select-positive .5 --gpu 2 --resample-timer 1 --tau .001 --log-interval 50 --reward-constant -2 --interaction-probability 0 --max-steps 200 --pretest-trials 20 --temporal-extend 300 --sac-alpha .2 --lookahead 4 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --max-distance-epsilon 6 --record-rollouts data/newrun/block --save-graph data/newrun/block_graph_model --save-interval 100 > logs/newrun/train_block_model.txt
 -->

python test_option.py --dataset-dir data/newrun/interaction_bbIv/ --graph-dir data/newrun/block_graph_model --object Block --buffer-len 500000 --gamma .99  --test-trials 10 --num-iters 5  --terminal-type proxist --reward-type proxist --sampler-type inst --option-type model --interaction-lambda 0 --parameterized-lambda 200 --true-reward-lambda 100 --epsilon-close .5 --time-cutoff 200 --temporal-extend 300 --reward-constant -1 --max-steps 5000 --print-test --use-interact --max-hindsight 12 --param-interaction --use-termination --param-first --change-option --gpu 3 --max-distance-epsilon 5 --visualize nosave



# 2DPushing domain
python generate_push_data.py data/pusher/pusher_random --num-frames 3000 --pushgripper

# train the gripper motion model
python construct_hypothesis_model.py --record-rollouts data/pusher/pusher_random/ --env 2DPushing --log-interval 500 --hidden-sizes 1024 1024 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 10000 --epsilon-schedule 3000 --num-iters 20000 --interaction-binary -1 -8 -16 --train --interaction-prediction .3 --base-variance 1e-4 --save-dir data/pusher/interaction_ag > logs/pusher/action_gripper_interaction.txt

# Run "test" to fill in parameter values for the hypothesis model
python construct_hypothesis_model.py --record-rollouts data/pusher/pusher_random --env 2DPushing --batch-size 64 --predict-dynamics --epsilon-schedule 1000 --action-shift --log-interval 1000 --num-frames 100000 --interaction-binary -1 -8 -16 --graph-dir data/action_graph/ --gpu 2  --dataset-dir data/pusher/interaction_ag --save-dir data/pusher/interaction_agv > logs/pusher/action_gripper_interactionv.txt

# train the gripper policy with HER and DQN
python train_option.py --dataset-dir data/pusher/interaction_agv/ --object Gripper --env 2DPushing --option-type model --buffer-len 500000 --num-steps 75 --gamma .99 --batch-size 16 --record-rollouts data/pusher/gripper --num-iters 4000 --terminal-type comb --reward-type comb --parameterized-lambda 30 --epsilon-close 1.5 --time-cutoff 75 --set-time-cutoff  --hidden-sizes 128 128 --learning-type herdqn --grad-epoch 75 --pretrain-iters 1000 --lr 1e-4 --epsilon .1 --epsilon-schedule 200 --select-positive .5 --gpu 1 --resample-timer 20 --tau 2500 --log-interval 25 --reward-constant -1 --max-steps 30 --print-test --pretest-trials 20 --hardcode-norm robopush 1 1 --interaction-probability 0 --interaction-prediction 0 --save-graph data/pusher/gripper_graph --save-interval 100 > logs/pusher/train_gripper.txt

# test gripper policy
python test_option.py --dataset-dir data/pusher/interaction_agv/ --env 2DPushing --graph-dir data/pusher/gripper_graph --object Gripper --buffer-len 500000 --gamma .99 --record-rollouts data/pusher/gripper_test --test-trials 20 --num-iters 10 --terminal-type comb --reward-type comb --parameterized-lambda 0 --epsilon-close 1 --time-cutoff 50 --set-time-cutoff --gpu 1 --reward-constant -1 --max-steps 30 --print-test > logs/pusher/test_gripper.txt

# train block motion model
python construct_hypothesis_model.py --env 2DPushing --record-rollouts data/pusher/gripper --hidden-sizes 512 512 --batch-size 64 --pretrain-iters 10000 --interaction-iters 50000 --epsilon-schedule 1000 --num-iters 100000 --posttrain-iters 0 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --graph-dir data/pusher/gripper_graph/ --train --gpu 0 --interaction-prediction .3 --predict-dynamics --action-shift  --interaction-weight 100 --save-dir data/pusher/interaction_gbI > logs/pusher/gripper_block_interaction.txt

# Test block motion model
python construct_hypothesis_model.py --record-rollouts data/pusher/gripper --env 2DPushing --batch-size 64 --predict-dynamics --epsilon-schedule 1000 --action-shift --log-interval 1000 --num-frames 100000 --interaction-binary -1 -8 -16 --graph-dir data/pusher/gripper_graph/ --gpu 0  --dataset-dir data/pusher/interaction_gbI --save-dir data/pusher/interaction_gbIv > logs/pusher/gripper_block_interactionv.txt

python construct_hypothesis_model.py --record-rollouts data/pusher/gripper --env 2DPushing --batch-size 64 --log-interval 1000 --num-frames 100000 --interaction-binary -1 -13 -13 --graph-dir data/pusher/gripper_graph/ --gpu 2  --dataset-dir data/pusher/interaction_gbI

# robosuite 2Dpushing domain
# generate the data
python generate_random_robopushing.py 1000 data/robopushing/random

# construct the forward model
python construct_hypothesis_model.py --record-rollouts data/robopushing/random/ --env RoboPushing --log-interval 500 --hidden-sizes 1024 1024 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 10000 --epsilon-schedule 3000 --num-iters 50000 --interaction-binary -1 -8 -16 --train --interaction-prediction .3 --norm-variance .025 --base-variance 1e-4 --model-error-significance .001 --save-dir data/robopushing/interaction_ag > logs/robopushing/action_gripper_interaction.txt

# test forward model
python construct_hypothesis_model.py --record-rollouts data/robopushing/random/ --env RoboPushing --batch-size 64 --predict-dynamics --epsilon-schedule 1000 --action-shift --log-interval 1000 --num-frames 100000 --interaction-binary -1 -8 -16 --norm-variance .025 --model-error-significance .5 --active-epsilon .01 --graph-dir data/action_graph/ --gpu 2  --dataset-dir data/robopushing/interaction_ag --save-dir data/robopushing/interaction_agv > logs/robopushing/action_gripper_interactionv.txt

# Train the gripper policy with HER and SAC
python train_option.py --dataset-dir data/robopushing/interaction_agv/ --object Gripper --env RoboPushing --option-type model --buffer-len 500000 --num-steps 100 --gamma .99 --batch-size 16 --num-iters 1000 --terminal-type tcomb --reward-type comb --parameterized-lambda 1 --epsilon-close .05 --time-cutoff 50 --hidden-sizes 64 64 64 --learning-type hersac --grad-epoch 20 --pretrain-iters 1000 --lr 1e-4 --epsilon 0.5 --select-positive .5 --gpu 1 --resample-timer 50 --tau .005 --log-interval 50 --reward-constant -1 --max-steps 50 --param-recycle .1 --sample-continuous 2 --print-test --pretest-trials 2 --relative-param 1 --sac-alpha -1 --no-input --observation-setting 1 0 0 1 1 0 0 0 0 --test-trials 1 --early-stopping 1 --pretest-trials 20 --record-rollouts data/robopushing/gripper --save-graph data/robopushing/gripper_graph --save-interval 100 > logs/robopushing/train_gripper.txt

python train_option.py --dataset-dir data/robopushing/interaction_agv/ --object Gripper --env RoboPushing --option-type model --buffer-len 500000 --num-steps 90 --gamma .99 --batch-size 16 --record-rollouts data/robopushing/gripper --num-iters 1000 --terminal-type comb --reward-type comb --parameterized-lambda 1 --epsilon-close 0.02 --time-cutoff 30  --hidden-sizes 128 128 --learning-type herddpg --grad-epoch 75 --pretrain-iters 1000 --lr 1e-4 --epsilon .1 --param-recycle .1 --sample-continuous 2 --epsilon-schedule 200 --select-positive .5 --gpu 1 --resample-timer 30 --tau .001 --log-interval 25 --reward-constant -1 --max-steps 30 --print-test --pretest-trials 20 --hardcode-norm robopush 1 1 --observation-setting 1 0 0 1 1 0 0 0 0 --interaction-probability 0 --interaction-prediction 0 --record-rollouts data/robopushing/gripper --save-graph data/robopushing/gripper_graph --save-interval 100 > logs/robopushing/train_gripper.txt


single step action distribution
learn one step-reachability: predict if a state is reachable within some number of time steps. Output all of the reachable raw states
self-consistency: 