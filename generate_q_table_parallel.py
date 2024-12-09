import numpy as np
from rubiks_cube_env import TwoByTwoCubeEnv
from cube_symmetries import CubeSymmetries, state_to_tuple
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

def get_reverse_action(action):
    """
    Returns the reverse action for a given action.
    
    Parameters:
    - action (int): The action index.
    
    Returns:
    - int: The reverse action index.
    """
    if action % 2 == 0:
        # Even actions are clockwise; reverse is counterclockwise (action +1)
        return action + 1
    else:
        # Odd actions are counterclockwise; reverse is clockwise (action -1)
        return action - 1

def expand_states(states, action_space):
    """
    Expands a list of states by applying all possible actions.
    
    Parameters:
    - states (list of np.ndarray): List of states to expand.
    - action_space (gym.Space): The action space of the environment.
    
    Returns:
    - list of tuples: Each tuple contains (new_state_tuple, reverse_action).
    """
    env = TwoByTwoCubeEnv()
    new_state_action_pairs = []
    
    for state in states:
        for action in range(action_space.n):
            env.set_state(state)
            next_state, _, done, _ = env.step(action)
            if done:
                # If the cube is solved, no need to add further actions
                continue
            next_state_tuple = state_to_tuple(next_state)
            reverse_action = get_reverse_action(action)
            new_state_action_pairs.append((next_state_tuple, reverse_action))
    
    return new_state_action_pairs

def generate_q_table_parallel(max_depth=10, num_processes=6):
    """
    Generates a Q-table for the 2x2x2 Rubik's Cube using parallel BFS.
    
    Parameters:
    - max_depth (int): The maximum number of moves to explore.
    - num_processes (int): Number of parallel processes to use.
    
    Returns:
    - dict: A dictionary mapping state tuples to optimal actions.
    """
    env = TwoByTwoCubeEnv()
    symmetries = CubeSymmetries()
    
    # Initialize the goal state
    goal_state = np.array([ [ [i, i], [i, i] ] for i in range(6) ])  # Shape: (6, 2, 2)
    env.set_state(goal_state)
    goal_state_tuple = state_to_tuple(goal_state)
    
    # Initialize BFS
    current_frontier = [goal_state]
    visited = set()
    visited.add(goal_state_tuple)
    q_table = {}  # state_tuple: optimal_action
    
    pool = Pool(processes=num_processes)
    
    for depth in range(max_depth):
        print(f"Exploring depth {depth + 1}/{max_depth} with {len(current_frontier)} states in frontier.")
        
        # Split the current frontier into chunks for each process
        chunk_size = len(current_frontier) // num_processes
        chunks = [current_frontier[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]
        # Handle any remaining states
        remainder = len(current_frontier) % num_processes
        for i in range(remainder):
            chunks[i].append(current_frontier[num_processes*chunk_size + i])
        
        # Define a partial function with fixed action_space
        expand_func = partial(expand_states, action_space=env.action_space)
        
        # Map the chunks to the pool
        results = pool.map(expand_func, chunks)
        
        # Flatten the list of results
        new_state_action_pairs = [item for sublist in results for item in sublist]
        
        # Initialize the next frontier
        next_frontier = []
        
        for state_tuple, reverse_action in new_state_action_pairs:
            if state_tuple not in visited:
                visited.add(state_tuple)
                q_table[state_tuple] = reverse_action
                # To prevent the next frontier from growing too large, consider only adding states up to a certain size
                next_frontier.append(state_tuple)
        
        current_frontier = [np.array(state).reshape((6,2,2)) for state in next_frontier]
    
    pool.close()
    pool.join()
    
    # Save the Q-table
    with open('q_table_parallel.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    
    print(f"Q-table generated with {len(q_table)} entries and saved to 'q_table_parallel.pkl'.")
    return q_table

if __name__ == "__main__":
    # Parameters
    MAX_DEPTH = 10          # Maximum number of moves from the goal state
    NUM_PROCESSES = 6      # Number of parallel processes (cores)
    
    # Generate the Q-table in parallel
    q_table = generate_q_table_parallel(max_depth=MAX_DEPTH, num_processes=NUM_PROCESSES)
