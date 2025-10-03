import torch
import torch.nn as nn
import torch.nn.functional as F
from agentes import Agent
from utils import *
import random
import numpy as np

# ==============================
#  ESTADO DEL JUEGO
# ==============================
class Connect4State:
    def __init__(self, n_rows=6, n_cols=7): 
        """
        Representa un snapshot del estado de Connect4.
        Contiene el tablero, el jugador actual, y si terminó o no.
        """
        self.board = create_board(n_rows, n_cols)  # tablero vacío
        self.current_player = 1                    # jugador que arranca
        self.isOver = False                        # flag de fin de juego
        self.winner = None                         # ganador (1, 2 o None)

    def copy(self):  
        """
        Crea una copia profunda del estado actual (nuevo objeto independiente).
        """
        new_state = Connect4State(self.board.shape[0], self.board.shape[1])
        new_state.board = np.copy(self.board)
        new_state.current_player = self.current_player
        new_state.isOver = self.isOver
        new_state.winner = self.winner
        return new_state

    def update_state(self):
        """
        Verifica si el juego terminó después de la última jugada,
        y actualiza current_player si no terminó.
        """
        is_over, winner_player = check_game_over(self.board)
        if is_over:
            self.isOver = True
            self.winner = winner_player
        else:
            # alterna el jugador
            self.current_player = 2 if self.current_player == 1 else 1

    def __eq__(self, other):
        """
        Compara si dos estados son iguales (para hashing o debug).
        """
        board_eq = np.array_equal(self.board, other.board)
        current_player_eq = self.current_player == other.current_player
        isOver_eq = self.isOver == other.isOver
        return board_eq and current_player_eq and isOver_eq

    def __hash__(self): 
        """
        Permite usar Connect4State como clave en diccionarios (Q-learning tabular).
        """
        return hash((tuple(self.board.flatten()), self.current_player, self.isOver))

    def __repr__(self):
        """
        Representación string para imprimir fácilmente el estado.
        """
        return str(self.board) + '\n' + str(self.current_player) + '\n' + str(self.isOver)


# ==============================
#  AMBIENTE
# ==============================
class Connect4Environment:
    def __init__(self, rows=6, cols=7):
        """
        Ambiente de Connect4: contiene el estado del juego y la lógica de transiciones.
        """
        self.rows = rows
        self.cols = cols
        self.game = Connect4State(self.rows, self.cols)
        self.done = False
        self.winner = None

    def reset(self):
        """
        Reinicia el juego a estado inicial y devuelve el estado.
        """
        self.game = Connect4State(self.rows, self.cols)
        self.done = False
        self.winner = None
        return self.game.copy()

    def available_actions(self):
        """
        Devuelve lista de columnas válidas (no llenas).
        """
        board = self.game.board
        available_actions = []
        for col in range(board.shape[1]):
            if board[0][col] == 0:
                available_actions.append(col)
        return available_actions

    def step(self, action):
        """
        Ejecuta una jugada (columna) y devuelve:
        (nuevo_estado, reward, done, info)
        """
        insert_token(self.game.board, action, self.game.current_player)

        # chequeo si terminó
        is_over, winner = check_game_over(self.game.board)

        if is_over:
            self.game.isOver = True
            self.game.winner = winner
            self.done = True
            self.winner = winner

            # asignación de recompensas
            if winner is None:   # empate
                reward = 0
            elif winner == self.game.current_player:
                reward = 1
            else:
                reward = -1

            return self.game.copy(), reward, True, {"winner": winner}

        else:
            # cambio de jugador
            self.game.current_player = 2 if self.game.current_player == 1 else 1
            return self.game.copy(), 0, False, {"winner": None}

    def render(self):
        """
        Imprime el tablero en consola.
        """
        board = self.game.board
        symbols = {0: ".", 1: "X", 2: "O"}
        print("\nTablero:")
        for row in board:
            print(" ".join(symbols[val] for val in row))
        print("0 1 2 3 4 5 6")
        print()


# ==============================
#  RED DQN
# ==============================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): 
        """
        Red neuronal simple: mapea estado -> Q-value por acción.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)   # capa oculta 1
        self.fc2 = nn.Linear(128, 128)         # capa oculta 2
        self.fc3 = nn.Linear(128, output_dim)  # salida (Q por cada acción posible)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # devuelve un vector de Q-values


# ==============================
#  AGENTE DQN
# ==============================
class DeepQLearningAgent:
    def __init__(self, state_shape, n_actions, device,
                 gamma, epsilon, epsilon_min, epsilon_decay,
                 lr, batch_size, memory_size, target_update_every): 
        """
        Implementación de un agente que usa Deep Q-Learning.
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_every = target_update_every

        # memoria de experiencias (FIFO)
        self.memory = []

        # dos redes: policy y target
        self.policy_net = DQN(state_shape[0] * state_shape[1], n_actions).to(device)
        self.target_net = DQN(state_shape[0] * state_shape[1], n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target no se entrena directamente

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
        self.episode_count = 0

    def preprocess(self, state):
        """
        Convierte Connect4State -> tensor torch listo para la red.
        """
        state_array = state.board.flatten()  
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        return state_tensor

    def select_action(self, state, valid_actions): 
        """
        Política epsilon-greedy:
        - Con prob epsilon: acción aleatoria válida.
        - Con prob (1-epsilon): acción con mayor Q-value entre las válidas.
        """
        sample = np.random.rand()
        if sample < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            state_tensor = self.preprocess(state)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            q_values = q_values.cpu().numpy().flatten()
            # filtrar solo acciones válidas
            q_values_valid = [(a, q_values[a]) for a in valid_actions]
            best_action = max(q_values_valid, key=lambda x: x[1])[0]
            return best_action

    def store_transition(self, s, a, r, s_next, done):
        """
        Guarda (estado, acción, recompensa, siguiente estado, done) en memoria.
        Si la memoria supera el límite, descarta la más vieja.
        """
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_next, done))

    def train_step(self): 
        """
        Muestra un batch de memoria y entrena la policy_net.
        """
        if len(self.memory) < self.batch_size:
            return None

        # sample aleatorio
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # preparar tensores
        states = torch.cat([self.preprocess(s) for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.cat([self.preprocess(s) for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Q-values de las acciones ejecutadas
        q_values = self.policy_net(states).gather(1, actions)

        # targets usando la target_net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # pérdida MSE
        loss = F.mse_loss(q_values, target_q_values)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # actualización de target_net cada N episodios
        if self.episode_count % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_epsilon(self):
        """
        Decaimiento de epsilon (exploración -> explotación).
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1


# ==============================
#  AGENTE ENTRENADO
# ==============================
class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu'):
        """
        Carga un agente DQN ya entrenado desde archivo.
        """
        self.device = device
        self.n_actions = n_actions
        self.policy_net = DQN(state_shape[0] * state_shape[1], n_actions).to(device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        self.policy_net.eval()

    def play(self, state, valid_actions): 
        """
        Selecciona la acción óptima (greedy) según la red entrenada.
        """
        state_array = state.board.flatten()
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        q_values = q_values.cpu().numpy().flatten()
        q_values_valid = [(a, q_values[a]) for a in valid_actions]
        best_action = max(q_values_valid, key=lambda x: x[1])[0]
        return best_action
