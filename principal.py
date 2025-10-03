import torch.nn as nn
from agentes import Agent
from utils import *

class Connect4State:
    def __init__(self,n_rows=6,n_cols=7): 
        """
        Inicializa el estado del juego Connect4.
        
        Args:
            Definir qué hace a un estado de Connect4.
        """
        self.board = create_board(n_rows, n_cols) #cambie que sea con atributos en ves d eun dict
        self.current_player = 1
        self.isOver = False
        self.winner = None

        pass

    def copy(self):  
        """
        Crea una copia profunda del estado actual.
        
        Returns:
            Una nueva instancia de Connect4State con los mismos valores.
        """
        new_state = Connect4State(self.board.shape[0], self.board.shape[1])
        new_state.board = np.copy(self.board)
        new_state.current_player = self.current_player
        new_state.isOver = self.isOver
        new_state.winner = self.winner
        return new_state

    def update_state(self):
        """
        Modifica las variables internas del estado luego de una jugada.

        Args:
            board 
            la ultima jugada 
            ... (_type_): _description_
            ... (_type_): _description_
        """
        is_over, winner_player = check_game_over(self.board) #lo cambie para que funcione con los atributos
        if is_over:
            self.isOver = True
            self.winner = winner_player
        else:
            self.current_player = 2 if self.current_player == 1 else 1

        

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        
        board_eq = np.array_equal(self.board, other.board)
        current_player_eq = self.current_player == other.current_player
        isOver_eq = self.isOver == other.isOver


        return board_eq and current_player_eq and isOver_eq
        

    def __hash__(self): 
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        return hash((tuple(self.board.flatten()), self.current_player, self.isOver))
        return result


    def __repr__(self):
        """
        Representación en string del estado.
        
        """
        return str(self.board) + '\n' + str(self.current_player) + '\n' + str(self.isOver)

class Connect4Environment:
    def __init__(self, rows=6, cols=7):
        """
        Inicializa el ambiente del juego Connect4.
        """
        self.rows = rows
        self.cols = cols
        self.game = Connect4State(self.rows, self.cols)
        self.done = False
        self.winner = None

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial.
        """
        self.game = Connect4State(self.rows, self.cols)
        self.done = False
        self.winner = None
        return self.game.copy()  # CAMBIO: devuelve copia del estado

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles).
        """
        board = self.game.board
        available_actions = []
        for col in range(board.shape[1]):  
            if board[0][col] == 0:
                available_actions.append(col)
        return available_actions

    def step(self, action):
        """
        Ejecuta una acción y devuelve (nuevo_estado, reward, done, info).
        """
        """
       # termina el jugador que empezo segundo, entonces la ultima jugada va a estar en el primero 
       # ultim ajugada: llena el tablero 
       # chequear si termino el juego psot hacer la jugada entonces nunca llega a que vuelva el primero 
        # si hay 42 jugada scual seria el que deberia check hgame is over primero 
        # fijarse si priemro hay que check game is over o si primero haces la jugada directamnete 
        is_over,winner = check_game_over(self.game.state)
        if(is_over):
            
            return{
                'nuevo_estado': 0,#pass,
                'reward': 1,
                'isOver': True,
                'winner': winner  
            }
        else:
            # Checkeamos que el action sea available-> asumimos que esta bien
            insert_token(self.game.state.board,action, self.game.state.current_player)
            #si lo es: lo ejecutamos chack
            
            #actualizamos el estado
            self.game.update_state()
            # devolvemos lo que nos pide
            """
        insert_token(self.game.board, action, self.game.current_player)

        # Chequear si terminó el juego
        is_over, winner = check_game_over(self.game.board)

        if is_over:
            self.game.isOver = True
            self.game.winner = winner
            self.done = True
            self.winner = winner

            if winner is None:   # empate
                reward = 0
            elif winner == self.game.current_player:
                reward = 1
            else:
                reward = -1

            return self.game.copy(), reward, True, {"winner": winner}

        else:
            # Si no terminó, cambiamos de jugador
            self.game.current_player = 2 if self.game.current_player == 1 else 1
            return self.game.copy(), 0, False, {"winner": None}

    def render(self):
        """
        Muestra el tablero en la consola.
        """
        board = self.game.board
        symbols = {0: ".", 1: "X", 2: "O"}
        print("\nTablero:")
        for row in board:
            print(" ".join(symbols[val] for val in row))
        print("0 1 2 3 4 5 6")
        print()


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): 
        """
        Inicializa la red neuronal DQN para el aprendizaje por refuerzo.
        
        Args:
            input_dim: Dimensión de entrada (número de features del estado).
            output_dim: Dimensión de salida (número de acciones posibles).
        """
        pass

    def forward(self, x):
        """
        Pasa la entrada a través de la red neuronal.
        
        Args:
            x: Tensor de entrada.
            
        Returns:
            Tensor de salida con los valores Q para cada acción.
        """
        pass

class DeepQLearningAgent:
    def __init__(self, state_shape, n_actions, device,
                 gamma, epsilon, epsilon_min, epsilon_decay,
                 lr, batch_size, memory_size, target_update_every): 
        """
        Inicializa el agente de aprendizaje por refuerzo DQN.
        
        Args:
            state_shape: Forma del estado (filas, columnas).
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación ('cpu' o 'cuda').
            gamma: Factor de descuento para recompensas futuras.
            epsilon: Probabilidad inicial de exploración.
            epsilon_min: Valor mínimo de epsilon.
            epsilon_decay: Factor de decaimiento de epsilon.
            lr: Tasa de aprendizaje.
            batch_size: Tamaño del batch para entrenamiento.
            memory_size: Tamaño máximo de la memoria de experiencias.
            target_update_every: Frecuencia de actualización de la red objetivo.
        """
        pass

    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.
        
        Args:
            state: Estado del juego.
            
        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        pass

    def select_action(self, state, valid_actions): 
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la acción seleccionada.
        """
        pass

    def store_transition(self, s, a, r, s_next, done):
        """
        Almacena una transición (estado, acción, recompensa, siguiente estado, terminado) en la memoria.
        
        Args:
            s: Estado actual.
            a: Acción tomada.
            r: Recompensa obtenida.
            s_next: Siguiente estado.
            done: Si el episodio terminó.
        """
        pass

    def train_step(self): 
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.
        
        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        pass

    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        pass

class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu'):
        """
        Inicializa un agente DQN pre-entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado.
            state_shape: Forma del estado del juego.
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación.
        """
        pass

    def play(self, state, valid_actions): 
        """
        Selecciona la mejor acción según el modelo entrenado.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la mejor acción según el modelo.
        """
        pass
