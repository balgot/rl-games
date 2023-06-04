from .tic_tac_toe import register_pyspiel as register_ttt
from .snakes import register_pyspiel as register_snakes

TTT_NAME = "ttt4x4-2"
register_ttt(4, 4, 2, TTT_NAME)

SNAKES_NAME = "snakes"
register_snakes(5, 5, SNAKES_NAME)
