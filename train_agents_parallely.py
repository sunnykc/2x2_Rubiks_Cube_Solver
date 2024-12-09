from q_learning_agent import QLearningAgent
from rubiks_cube_env import TwoByTwoCubeEnv

import numpy as np
import random
import multiprocessing
import math
import pickle
import time
import os

# Training parameters
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
EPISODES = 2
SCRAMBLE_DEPTH = 6
NUM_WORKERS = 6  # For a 6-core machine

def print_cube(state):
    face_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'TOP', 'BOTTOM']
    colors = {0: 'W', 1: 'Y', 2: 'B', 3: 'G', 4: 'R', 5: 'O'}
    for idx, face in enumerate(state):
        readable_face = [[colors[sticker] for sticker in row] for row in face]
        print(f"{face_names[idx]}: {readable_face}")
    print()

def cube_to_string(state):
    """
    Convert state to a string representation for logging.
    """
    return str(state.tolist())

def generate_scramble_sequence(env, scramble_depth=SCRAMBLE_DEPTH):
    state = env.reset()
    for _ in range(scramble_depth):
        action = env.action_space.sample()
        state, _, _, _ = env.step(action)
    return state

def worker_run_episodes(worker_id, episodes, q_table_data, epsilon, epsilon_decay, learning_rate, discount_factor, scramble_depth):
    env = TwoByTwoCubeEnv()
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=1.0  # We'll decay epsilon in the main process
    )
    agent.q_table = q_table_data

    log_filename = f"worker_{worker_id}_log.txt"
    if os.path.exists(log_filename):
        os.remove(log_filename)

    total_reward_sum = 0
    with open(log_filename, "w") as log_file:
        log_file.write(f"Worker {worker_id} starting {episodes} episodes.\n")

        for ep in range(episodes):
            state = generate_scramble_sequence(env, scramble_depth)
            done = False
            ep_step = 0
            log_file.write(f"Episode {ep+1}/{episodes} started. Scrambled state: {cube_to_string(state)}\n")

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                # Log step details
                log_file.write(f"  Step {ep_step+1}: State: {cube_to_string(state)}, Action: {action}, "
                               f"Reward: {reward}, Done: {done}\n")

                agent.update_q_table(state, action, reward, next_state, done)
                state = next_state
                ep_step += 1
            total_reward_sum += reward
        log_file.write(f"Worker {worker_id} finished.\n")

    return agent.q_table, total_reward_sum

def merge_q_tables(q_tables):
    merged_q_table = {}
    count_table = {}

    for qtab in q_tables:
        for key, value in qtab.items():
            if key not in merged_q_table:
                merged_q_table[key] = value
                count_table[key] = 1
            else:
                merged_q_table[key] += value
                count_table[key] += 1

    for key in merged_q_table:
        merged_q_table[key] /= count_table[key]

    return merged_q_table

def q_table_statistics(q_table):
    """
    Compute and return statistics about the Q-table: number of entries, average Q-value.
    """
    if len(q_table) == 0:
        return 0, 0.0
    values = list(q_table.values())
    avg_val = sum(values) / len(values)
    return len(q_table), avg_val

def train_agent_parallel():
    start_time = time.time()

    base_env = TwoByTwoCubeEnv()
    base_agent = QLearningAgent(
        action_space=base_env.action_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY
    )

    episodes_per_batch = 10 * NUM_WORKERS
    num_batches = math.ceil(EPISODES / episodes_per_batch)

    current_epsilon = INITIAL_EPSILON
    q_table_data = {}

    for batch_idx in range(num_batches):
        episodes_this_batch = min(episodes_per_batch, EPISODES - batch_idx * episodes_per_batch)
        episodes_per_worker = episodes_this_batch // NUM_WORKERS
        remainder = episodes_this_batch % NUM_WORKERS

        tasks = [episodes_per_worker] * NUM_WORKERS
        for i in range(remainder):
            tasks[i] += 1

        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = []
            for w in range(NUM_WORKERS):
                result = pool.apply_async(
                    worker_run_episodes,
                    args=(w, tasks[w], q_table_data, current_epsilon, 1.0, LEARNING_RATE, DISCOUNT_FACTOR, SCRAMBLE_DEPTH)
                )
                results.append(result)

            q_tables = []
            total_rewards = 0
            for res in results:
                qtab, tr = res.get()
                q_tables.append(qtab)
                total_rewards += tr

        if len(q_tables) > 0:
            q_table_data = merge_q_tables(q_tables)

        # Decay epsilon
        for _ in range(episodes_this_batch):
            if current_epsilon > base_agent.epsilon_min:
                current_epsilon *= EPSILON_DECAY

        avg_reward = total_rewards / episodes_this_batch if episodes_this_batch > 0 else 0.0
        print(f"Batch {batch_idx+1}/{num_batches} - Episodes: {(batch_idx+1)*episodes_per_batch}, Avg Reward: {avg_reward:.2f}, Epsilon: {current_epsilon:.4f}")

    # Save the final Q-table
    with open('q_table_2x2x2.pkl', 'wb') as f:
        pickle.dump(q_table_data, f)

    end_time = time.time()
    training_duration = end_time - start_time

    # Print Q-table statistics
    size, avg_val = q_table_statistics(q_table_data)
    print("Parallel training complete.")
    print(f"Training time: {training_duration:.2f} seconds")
    print(f"Q-table saved to 'q_table_2x2x2.pkl'. Q-table size: {size}, Average Q-value: {avg_val:.4f}")

    env = TwoByTwoCubeEnv()
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=0.0,  # no exploration needed for demonstration
        epsilon_decay=1.0
    )
    agent.q_table = q_table_data
    return agent, env

def demonstrate(agent, env):
    start_demo_time = time.time()

    # Demonstration log file
    demo_log_file = "demonstration_steps_log.txt"
    if os.path.exists(demo_log_file):
        os.remove(demo_log_file)

    with open(demo_log_file, "w") as log_file:
        log_file.write("Starting demonstration...\n")

        print("Generating a scrambled cube:")
        scrambled_state = generate_scramble_sequence(env, SCRAMBLE_DEPTH)
        print_cube(scrambled_state)
        log_file.write(f"Scrambled State: {cube_to_string(scrambled_state)}\n")

        print("Solving the scrambled cube:")
        state = scrambled_state
        done = False
        step = 0
        actions_taken = []

        while not done:
            action = agent.choose_action(state)
            next_state, _, done, _ = env.step(action)
            step += 1
            actions_taken.append(action)

            # Log each step in demonstration
            log_file.write(f"Step {step}: State: {cube_to_string(state)}, Action: {action}, Done: {done}\n")

            print(f"Step {step}: Action {action}")
            print_cube(next_state)
            state = next_state

        end_demo_time = time.time()
        demo_duration = end_demo_time - start_demo_time

        # Demonstration summary
        log_file.write(f"Demonstration completed in {demo_duration:.2f} seconds.\n")
        log_file.write(f"Number of steps taken: {step}\n")
        log_file.write("Actions taken: " + str(actions_taken) + "\n")

    print("Cube solved!")
    print(f"Demonstration completed in {demo_duration:.2f} seconds.")
    print(f"Number of steps taken: {step}")
    print("Actions taken during demonstration:", actions_taken)

if __name__ == "__main__":
    agent, env = train_agent_parallel()
    demonstrate(agent, env)
