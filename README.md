requirements: pytorch, opencv, imageio

python generate_breakout_data.py data/random 3000

python construct_dataset_model.py --graph-dir data/graph/ --dataset-dir data/random/ --num-frames 1000 --object Action
python add_option.py --dataset-dir data/random/ --object Paddle --buffer-steps 5 --num-steps 5 --factor 8 --learning-type a2c --gamma .1 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --record-rollouts data/paddle --num-iters 10000 --save-graph data/tempgraph --train --save-interval 1000 > paddletrain.txt

python construct_dataset_model.py --graph-dir data/paddlegraph/ --dataset-dir data/paddle/ --num-frames 10000 --object Paddle --target Ball

python add_option.py --dataset-dir data/paddle/ --object Ball --buffer-steps 5 --num-steps 5 --factor 32 --learning-type a2c --gamma .99 --batch-size 5 --gpu 2 --normalize --entropy-coef .01 --num-iters 100000 --save-interval 1000 --save-graph data/tempgraph --set-time-cutoff --graph-dir data/paddlegraph --record-rollouts data/Ball

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


New interaction model
Train paddle model
python construct_hypothesis_model.py --dataset-dir data/random/ --log-interval 500 --factor 32 --num-layers 2 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 0 --interaction-iters 100 --num-iters 3000

python construct_hypothesis_model.py --dataset-dir data/random/ --log-interval 500 --factor 8 --num-layers 2 --predict-dynamics --action-shift --batch-size 10 --pretrain-iters 0 --epsilon-schedule 4000 --num-iters 16000 --interaction-binary -1 -8 --save-dir data/interaction_ln > train_hypothesis.txt

python add_option.py --dataset-dir data/random/ --object Paddle --option-type model --buffer-steps 5 --num-steps 5 --gamma .1 --batch-size 5 --record-rollouts data/paddle --num-iters 1000

python add_option.py --dataset-dir data/interaction_ln/ --object Paddle --option-type model --buffer-steps 10 --num-steps 10 --gamma .99 --batch-size 10 --record-rollouts data/paddle --num-iters 100000 --terminal-type comb --reward-type comb --parameterized-lambda 0.1 --epsilon-close 1 --time-cutoff 100 --set-time-cutoff --init-form xnorm --train --normalize --num-layers 2 --factor 8 > train2.txt