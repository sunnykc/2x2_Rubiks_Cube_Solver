# demonstrate_agent.py

import numpy as np
from rubiks_cube_env import TwoByTwoCubeEnv
from q_learning_agent import QLearningAgent
from cube_symmetries import CubeSymmetries, state_to_tuple
import pickle
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Define color display mapping
COLOR_DISPLAY = {
    'W': Fore.WHITE + 'W' + Style.RESET_ALL,
    'Y': Fore.YELLOW + 'Y' + Style.RESET_ALL,
    'B': Fore.BLUE + 'B' + Style.RESET_ALL,
    'G': Fore.GREEN + 'G' + Style.RESET_ALL,
    'R': Fore.RED + 'R' + Style.RESET_ALL,
    'O': Fore.MAGENTA + 'O' + Style.RESET_ALL  # Using MAGENTA for Orange
}

# Define action descriptions
ACTION_DESCRIPTIONS = {
    0: "Front Clockwise",
    1: "Front Counterclockwise",
    2: "Back Clockwise",
    3: "Back Counterclockwise",
    4: "Left Clockwise",
    5: "Left Counterclockwise",
    6: "Right Clockwise",
    7: "Right Counterclockwise",
    8: "Up Clockwise",
    9: "Up Counterclockwise",
    10: "Down Clockwise",
    11: "Down Counterclockwise"
}

def state_to_tuple(state):
    return tuple(state.flatten())

def int_to_color(state_int):
    colors = ['W', 'Y', 'B', 'G', 'R', 'O']
    return np.vectorize(lambda x: colors[x])(state_int)

def print_cube(state_int):
    state = int_to_color(state_int)
    print("       ", " ".join(COLOR_DISPLAY[sticker] for sticker in state[4][0]))  # UP top row
    print("       ", " ".join(COLOR_DISPLAY[sticker] for sticker in state[4][1]))  # UP bottom row
    print(" ".join(COLOR_DISPLAY[sticker] for sticker in state[2][0]),
          " ".join(COLOR_DISPLAY[sticker] for sticker in state[0][0]),
          " ".join(COLOR_DISPLAY[sticker] for sticker in state[3][0]))  # LEFT, FRONT, RIGHT top rows
    print(" ".join(COLOR_DISPLAY[sticker] for sticker in state[2][1]),
          " ".join(COLOR_DISPLAY[sticker] for sticker in state[0][1]),
          " ".join(COLOR_DISPLAY[sticker] for sticker in state[3][1]))  # LEFT, FRONT, RIGHT bottom rows
    print("       ", " ".join(COLOR_DISPLAY[sticker] for sticker in state[1][0]))  # BACK top row
    print("       ", " ".join(COLOR_DISPLAY[sticker] for sticker in state[1][1]))  # BACK bottom row
    print("       ", " ".join(COLOR_DISPLAY[sticker] for sticker in state[5][0]))  # DOWN top row
    print("       ", " ".join(COLOR_DISPLAY[sticker] for sticker in state[5][1]))  # DOWN bottom row
    print()

def load_q_table(filename='q_table_trained.pkl'):
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
    print(f"Q-table loaded from '{filename}'.")
    return q_table

def demonstrate(agent, env, max_steps=20):
    """
    Demonstrates the agent attempting to solve a scrambled cube.
    
    Parameters:
    - agent (QLearningAgent): The trained Q-learning agent.
    - env (TwoByTwoCubeEnv): The cube environment.
    - max_steps (int): Maximum number of steps to attempt.
    """
    state = env.reset()
    print("Initial Scrambled Cube:")
    print_cube(state)
    
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        action = agent.choose_action(state, exploit=True)  # Exploit the learned policy
        next_state, reward, done, _ = env.step(action)
        print(f"Step {steps + 1}: Action {action} ({ACTION_DESCRIPTIONS.get(action, 'Unknown Action')})")
        print_cube(next_state)
        state = next_state
        steps += 1
    
    if env._is_solved():
        print(f"🎉 Cube solved in {steps} steps!")
    else:
        print(f"❌ Failed to solve the cube within {max_steps} steps.")

def main():
    # Initialize environment and agent
    env = TwoByTwoCubeEnv()
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.0,  # Set to 0 to fully exploit the learned policy
        epsilon_decay=1.0,
        min_epsilon=0.0,
        replay_buffer_size=50000,
        batch_size=64
    )
    
    # Load the trained Q-table
    agent.load_q_table('q_table_trained.pkl')
    
    # Demonstrate solving the cube
    demonstrate(agent, env, max_steps=200)

if __name__ == "__main__":
    main()
