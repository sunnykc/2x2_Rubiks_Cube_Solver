import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNRubiksCubeAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99):
        """
        Initializes the DQN agent for a 2x2x2 Rubik's Cube solver.

        Parameters:
        - state_size: Size of the flattened state (24 for 2x2x2 cube).
        - action_size: Number of possible actions (12 for 6 faces × 2 directions).
        - learning_rate: Learning rate for the optimizer.
        - discount_factor: Gamma, the discount factor for future rewards.
        """
        self.state_size = state_size  # For 2x2x2: 24
        self.action_size = action_size  # For 2x2x2: 12
        self.memory = deque(maxlen=20000)  # Replay memory
        self.gamma = discount_factor  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.learning_rate = learning_rate  # Learning rate for optimizer
        self.batch_size = 64  # Batch size for replay
        self.model = self._build_model()  # Main Q-network
        self.target_model = self._build_model()  # Target Q-network
        self.update_target_model()  # Synchronize weights initially

    def _build_model(self):
        """
        Builds the neural network model for approximating Q-values.

        Returns:
        - Compiled Keras Sequential model.
        """
        model = models.Sequential()
        model.add(layers.Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies the weights of the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay memory.

        Parameters:
        - state: Current state.
        - action: Action taken.
        - reward: Reward received.
        - next_state: Next state after taking the action.
        - done: Boolean indicating if the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses an action based on the current policy (epsilon-greedy).

        Parameters:
        - state: The current state.

        Returns:
        - Action index.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)  # Exploit
        return np.argmax(q_values[0])

    def replay(self):
        """
        Trains the model using randomly sampled transitions from the replay memory.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples for a batch
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target += self.gamma * np.amax(t)  # Bellman equation
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target  # Update the Q-value for the action
            states[i] = state
            targets[i] = target_f[0]

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
