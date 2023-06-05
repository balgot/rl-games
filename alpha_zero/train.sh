#!/bin/bash

echo "training tic tac toe models..."
python train.py --ttt --lr 0.001 --max-steps 100 --max-simulations 50 --batch-size 64 --checkpoint-dir "./logs-ttt" --wandbproject "pv056-tic-tac-toe"  --wandbname "mlp-big-50"
python train.py --ttt --lr 0.001 --max-steps 30 --max-simulations 50 --batch-size 64 --checkpoint-dir "./logs-ttt" --wandbproject "pv056-tic-tac-toe" --wandbname "cnn-50" --model conv2d --nn-width 64 --nn-depth 4
python train.py --ttt --lr 0.001 --max-steps 100 --max-simulations 20 --batch-size 64 --checkpoint-dir "./logs-ttt" --wandbproject "pv056-tic-tac-toe"  --wandbname "mlp-mid-20" --nn-width 64 --nn-depth 4


echo "training snakes models..."
python train.py --snakes --lr 0.001 --max-steps 100 --max-simulations 50 --batch-size 64 --checkpoint-dir "./logs-snake" --wandbproject "pv056-snakes"  --wandbname "mlp-big-50"
python train.py --snakes --lr 0.001 --max-steps 30 --max-simulations 50 --batch-size 64 --checkpoint-dir "./logs-snake" --wandbproject "pv056-snakes" --wandbname "cnn-50" --model conv2d --nn-width 64 --nn-depth 4
python train.py --snakes --lr 0.001 --max-steps 100 --max-simulations 20 --batch-size 64 --checkpoint-dir "./logs-snake" --wandbproject "pv056-snakes"  --wandbname "mlp-mid-20" --nn-width 64 --nn-depth 4


echo "training nim models..."
python train.py --cards --lr 0.001 --max-steps 100 --max-simulations 50 --batch-size 64 --checkpoint-dir "./logs-nim" --wandbproject "pv056-nim"  --wandbname "mlp-big-50"
python train.py --cards --lr 0.0001 --max-steps 100 --max-simulations 20 --batch-size 256 --checkpoint-dir "./logs-nim" --wandbproject "pv056-nim"  --wandbname "mlp-mid-20" --nn-width 64 --nn-depth 2
