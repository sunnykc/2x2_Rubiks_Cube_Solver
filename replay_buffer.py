from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Initializes the replay buffer.
        
        Parameters:
        - capacity (int): Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        """
        Adds an experience to the buffer.
        
        Parameters:
        - experience (tuple): A tuple of (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size=64):
        """
        Samples a random batch of experiences from the buffer.
        
        Parameters:
        - batch_size (int): Number of experiences to sample.
        
        Returns:
        - list of tuples: A list of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
