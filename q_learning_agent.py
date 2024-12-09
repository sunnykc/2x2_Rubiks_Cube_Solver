import numpy as np
import pickle
from replay_buffer import ReplayBuffer
from cube_symmetries import CubeSymmetries, state_to_tuple

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                 replay_buffer_size=10000, batch_size=64):
        """
        Initializes the Q-learning agent.
        
        Parameters:
        - action_space (gym.Space): The action space of the environment.
        - learning_rate (float): Learning rate (alpha).
        - discount_factor (float): Discount factor (gamma).
        - epsilon (float): Initial exploration rate.
        - epsilon_decay (float): Decay rate for exploration.
        - min_epsilon (float): Minimum exploration rate.
        - replay_buffer_size (int): Capacity of the replay buffer.
        - batch_size (int): Number of experiences to sample for each training step.
        """
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        self.batch_size = batch_size
        self.q_table = {}  # key: state_tuple, value: np.array of Q-values
        self.symmetries = CubeSymmetries()
    
    def choose_action(self, state, exploit=False):
        """
        Chooses an action based on the epsilon-greedy policy.
        
        Parameters:
        - state (np.ndarray): The current state of the environment.
        - exploit (bool): If True, choose the best action without exploration.
        
        Returns:
        - int: The chosen action.
        """
        canonical_state = self.symmetries.get_canonical_state(state)
        if canonical_state not in self.q_table:
            self.q_table[canonical_state] = np.zeros(self.action_space.n)
        
        if exploit or np.random.rand() > self.epsilon:
            # Exploit: Choose the action with the highest Q-value
            action = np.argmax(self.q_table[canonical_state])
        else:
            # Explore: Choose a random action
            action = self.action_space.sample()
        return action
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Updates the Q-table based on the experience.
        
        Parameters:
        - state (np.ndarray): The previous state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.ndarray): The resulting state.
        - done (bool): Whether the episode has ended.
        """
        canonical_state = self.symmetries.get_canonical_state(state)
        canonical_next_state = self.symmetries.get_canonical_state(next_state)
        
        if canonical_state not in self.q_table:
            self.q_table[canonical_state] = np.zeros(self.action_space.n)
        
        if canonical_next_state not in self.q_table:
            self.q_table[canonical_next_state] = np.zeros(self.action_space.n)
        
        q_current = self.q_table[canonical_state][action]
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[canonical_next_state])
        
        # Q-learning update rule
        self.q_table[canonical_state][action] += self.lr * (q_target - q_current)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        
        Parameters:
        - state (np.ndarray): The previous state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.ndarray): The resulting state.
        - done (bool): Whether the episode has ended.
        """
        self.replay_buffer.add((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Samples a batch of experiences from the replay buffer and performs Q-table updates.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to replay
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_table(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """
        Decays the exploration rate epsilon.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
    
    def save_q_table(self, filename='q_table.pkl'):
        """
        Saves the Q-table to a file.
        
        Parameters:
        - filename (str): The filename to save the Q-table.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to '{filename}'.")
    
    def load_q_table(self, filename='q_table.pkl'):
        """
        Loads the Q-table from a file.
        
        Parameters:
        - filename (str): The filename from which to load the Q-table.
        """
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from '{filename}'.")
        print(f"Random Q-table loaded with {len(self.q_table)} states and saved to 'q_table_random.pkl'.")


