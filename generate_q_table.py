# generate_random_q_table.py

import numpy as np
from rubiks_cube_env import TwoByTwoCubeEnv
import pickle
from collections import deque

def state_to_tuple(state):
    """
    Convert the state array to a tuple for use as a dictionary key.
    
    Parameters:
    - state (np.ndarray): The cube state as integer indices.
    
    Returns:
    - tuple: A tuple representation of the state.
    """
    return tuple(state.flatten())

def generate_random_q_table(max_depth=10, sample_size=None):
    """
    Generates a Q-table for the 2x2x2 Rubik's Cube by performing BFS from the goal state
    and assigns random Q-values to each action for every state.
    
    Parameters:
    - max_depth (int): The maximum number of moves to explore from the goal state.
    - sample_size (int, optional): If set, limits the Q-table to this number of randomly sampled states.
    
    Returns:
    - dict: A dictionary mapping state tuples to numpy arrays of random Q-values for each action.
    """
    env = TwoByTwoCubeEnv()
    
    # Initialize the goal state
    goal_state = np.array([ [ [i, i], [i, i] ] for i in range(6) ])  # Shape: (6, 2, 2)
    env.set_state(goal_state)
    
    goal_state_tuple = state_to_tuple(goal_state)
    
    # Initialize BFS
    queue = deque()
    queue.append((goal_state, 0))  # (state, depth)
    
    visited = {goal_state_tuple}
    q_table = {}  # state_tuple: Q-values array
    
    while queue:
        current_state, depth = queue.popleft()
        current_state_tuple = state_to_tuple(current_state)
        
        if sample_size and len(q_table) >= sample_size:
            break  # Reached the desired sample size
        
        if current_state_tuple not in q_table:
            # Assign random Q-values for each action
            q_values = np.random.uniform(low=-1, high=1, size=env.action_space.n)
            q_table[current_state_tuple] = q_values
        
        if depth >= max_depth:
            continue  # Limit the depth to prevent excessive computation
        
        for action in range(env.action_space.n):
            # Apply the action to get the next state
            env.set_state(current_state)
            next_state, _, _, _ = env.step(action)
            next_state_tuple = state_to_tuple(next_state)
            
            if next_state_tuple not in visited:
                visited.add(next_state_tuple)
                queue.append((next_state, depth + 1))
    
    # Optionally, limit the Q-table to a sample size
    if sample_size and len(q_table) > sample_size:
        # Randomly sample the desired number of states
        sampled_keys = np.random.choice(list(q_table.keys()), size=sample_size, replace=False)
        q_table = {key: q_table[key] for key in sampled_keys}
    
    # Save the Q-table
    with open('q_table_random.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    
    print(f"Random Q-table generated with {len(q_table)} states and saved to 'q_table_random.pkl'.")
    return q_table

if __name__ == "__main__":
    # Parameters
    MAX_DEPTH = 10          # Maximum number of moves from the goal state
    SAMPLE_SIZE = None      # Set to an integer to limit the number of states
    
    # Generate the random Q-table
    q_table = generate_random_q_table(max_depth=MAX_DEPTH, sample_size=SAMPLE_SIZE)
