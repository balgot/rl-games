# Snakes

2 player game of snake. Rules:
* game is played by two players
* each step, both players make a move (so not sequential, but can be viewed
as such)
* possible actions: UP, DOWN, LEFT, RIGHT
* each step, if there is none, a "fruit" is spawned randomly on the game plan
* upon reaching the fruit, snake gets longer
* game ends when one of the snakes crashes to wall, itself or other snake
* rewards: 1 for win (enemy kills themselves), 0 (both kill themselves at
the same time)
* both players spawn one tile long, on the opposite ends of the plan, zero
velocities