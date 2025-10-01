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
        current_player = 1
        board = create_board(n_rows,n_cols)
        self.state = {
            'board': board,
            'current_player' : current_player,
            'isOver': False,
            'winner': None
        }
        
        
        
        pass

    def copy(self):  
        """
        Crea una copia profunda del estado actual.
        
        Returns:
            Una nueva instancia de Connect4State con los mismos valores.
        """
        return np.copy(self.state)

    def update_state(self):
        """
        Modifica las variables internas del estado luego de una jugada.

        Args:
            board 
            la ultima jugada 
            ... (_type_): _description_
            ... (_type_): _description_
        """
        is_over, winner_player = check_game_over(self.state.board)
        if(is_over):
            self.state.isOver = True
            self.state.winner = winner_player
        else:
            if(self.state.current_player == 1):
              self.state.current_player = 2  
            else:
              self.state.current_player = 1
        

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        
        board_eq = np.array_equal(self.state.board, other.state.board)
        current_player_eq = self.state.current_player == other.state.current_player
        isOver_eq = self.state.isOver == other.state.isOver

        return board_eq and current_player_eq and isOver_eq
        

    def __hash__(self): 
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        input_hash = str(self.state.board)+str(self.state.current_player)+str(self.state.isOver)
        result = hash_function(input_hash) # To do
        return result


    def __repr__(self):
        """
        Representación en string del estado.
        
        """
        return str(self.state.board)+'\n'+str(self.state.current_player)+'\n'+str(self.state.isOver)

class Connect4Environment:
    def __init__(self):
        """
        Inicializa el ambiente del juego Connect4.
        
        Args:
            1 o 2 jugadores Definir las variables de instancia de un ambiente de Connect4

        """        
        self.rows = 6
        self.cols = 7
        self.game = Connect4State(self.rows,self.cols)
        # game.state

        self.done = False
        self.winner = None

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.
        
        """
        self.rows = 6
        self.cols = 7
        self.game = Connect4State(self.rows,self.cols)
        # game.state

        self.done = False
        self.winner = None

        return self.game.state
        

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.
        
        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        board = self.game.state.board
        available_actions = np.array([])
        for col in board.shape[1]: # num de columnas
            if(board[0][col] == 0):
                np.append(available_actions,col)
        return available_actions
    

    def step(self, action):
        """
        Ejecuta una acción.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Devuelve la tupla: nuevo_estado, reward, terminó_el_juego?, ganador
        Si terminó_el_juego==false, entonces ganador es None.
        
        Args:
            action: Acción elegida por un agente.
            
        """
       #
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
            
            
        
        pass

    def render(self):
        """
        Muestra visualmente el estado actual del tablero en la consola.

        """
        pass

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
                 lr, batch_size, memory_size target_update_every): 
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
