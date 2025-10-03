from principal import Connect4Environment
from utils import insert_token, check_game_over
import numpy as np

env = Connect4Environment()
state = env.reset()
done = False

while not done:
    actions = env.available_actions()
    action = np.random.choice(actions)  # jugada aleatoria
    next_state, reward, done, info = env.step(action)
    env.render()
    state = next_state

print("Partida terminada. Ganador:", info["winner"])
