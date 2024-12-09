# train_q_learning_agent.py

import numpy as np
from rubiks_cube_env import TwoByTwoCubeEnv
from q_learning_agent import QLearningAgent
from cube_symmetries import CubeSymmetries
import pickle, time

def train_agent():
    # Hyperparameters
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 20  # To prevent excessively long episodes
    
    # Initialize environment and agent
    env = TwoByTwoCubeEnv()
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
        replay_buffer_size=50000,
        batch_size=64
    )
    
    # Optionally, load a pre-trained Q-table
    # agent.load_q_table('q_table.pkl')
    
    total_start_time = time.time()  # Start tracking the total time


    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.replay()
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        
        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode}/{EPISODES} - Steps: {steps} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.4f}")
    
    total_end_time = time.time()  # End tracking the total time
    total_elapsed_time = total_end_time - total_start_time  # Calculate total elapsed time

    # Save the trained Q-table
    agent.save_q_table('q_table_trained.pkl')
    print("Training completed and Q-table saved.")
    print(f"Total training time: {total_elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    train_agent()
