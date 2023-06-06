# Implemented Games

This folder contains implementation of games:
1. 2-player snakes
2. generalised tic-tac-toe


## Installation

```bash
pip install PATH_TO_THIS_FOLDER
```


## Usage

For both games, the interface is same:
1. import necessary functions, e.g. `from games import snakes`
2. to use with pyspiel:

```python
snakes.register_pyspiel(HEIGHT, WIDTH, "my_snakes_game")
game = pyspiel.load_game("my_snakes_game")
```

3. you can also play this game visually (using pygame):

```python
# register pyspiel as above
snakes.play_pygame(game_name, PLAYER, PLAYER2)
```
