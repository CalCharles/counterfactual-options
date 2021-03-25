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


conda create -n cv python=3.7.9
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install imageio
pip install opencv-python

python add_option.py --object Raw --option-type raw --true-environment --env Pend-Gym --buffer-steps 500000 --num-steps 1 --gamma .99 --batch-size 128 --num-iters 20000 --terminal-type true --reward-type true --epsilon-close 1 --init-form none --train --normalize --policy-type actorcritic --learning-type ddpg --grad-epoch 5 --warm-up 128 --warm-update 0 --lr 1e-4 --epsilon 0 --behavior-type greedyQ --return-form none --Q-critic --gpu 2 --log-interval 200 --optim Adam --factor 8 --num-layers 2 --use-layer-norm --double-Q .001 --actor-critic-optimizer