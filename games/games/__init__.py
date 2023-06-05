from .tic_tac_toe import register_pyspiel as register_ttt
from .snakes import register_pyspiel as register_snakes

TTT_NAME = "ttt"
register_ttt(5, 5, 3, TTT_NAME)

SNAKES_NAME = "snakes"
register_snakes(5, 5, SNAKES_NAME)
