#!/bin/bash

mkdir -p eval/ttt
mkdir -p eval/snakes
mkdir -p eval/nim

python evaluate.py --ttt --random --path eval/ttt/random.csv
python evaluate.py --ttt --trained --path eval/ttt/mlp-big.csv --runname ttt-mlp-big-50 --logs "logs-ttt/" --id "miba/pv056-tic-tac-toe/gkl0i1gj" --checkpoint "-1"
python evaluate.py --ttt --trained --path eval/ttt/cnn.csv --runname ttt-cnn-50 --logs "logs-ttt/" --id "miba/pv056-tic-tac-toe/r3foj5me" --checkpoint "-1"
python evaluate.py --ttt --trained --path eval/ttt/mlp-mid.csv --runname ttt-mlp-mid-20 --logs "logs-ttt/" --id "miba/pv056-tic-tac-toe/7l25tadb" --checkpoint "-1"

python evaluate.py --snakes --random --path eval/snakes/random.csv
python evaluate.py --snakes --trained --path eval/snakes/mlp-big.csv --runname snake-mlp-big-50 --logs "logs-snake/" --id "miba/pv056-snakes/xyc229ml" --checkpoint "-1"
python evaluate.py --snakes --trained --path eval/snakes/cnn.csv --runname snake-cnn-50 --logs "logs-snake/" --id "miba/pv056-snakes/bcru9zz4" --checkpoint "-1"
python evaluate.py --snakes --trained --path eval/snakes/mlp-mid.csv --runname snake-mlp-mid-20 --logs "logs-snake/" --id "miba/pv056-snakes/8xdmyrjz" --checkpoint "-1"

python evaluate.py --cards --random --path eval/nim/random.csv
python evaluate.py --cards --trained --path eval/nim/mlp-big.csv --runname nim-mlp-big-50 --logs "logs-nim/" --id "miba/pv056-nim/hlg1za2b" --checkpoint "-1"
python evaluate.py --cards --trained --path eval/nim/mlp-mid.csv --runname nim-mlp-mid-20 --logs "logs-nim/" --id "miba/pv056-nim/07glb1q9" --checkpoint "-1"
