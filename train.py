from q_learning_agent import QLearningAgent
from rubiks_cube_env import TwoByTwoCubeEnv  # Assuming the 2x2x2 environment is defined here
import numpy as np


def generate_scramble_sequence(env, scramble_depth=3):
    """
    Generate a sequence of moves to systematically scramble the cube.

    Parameters:
    - env: The 2x2x2 Rubik's Cube environment.
    - scramble_depth: Number of moves in the scramble sequence.

    Returns:
    - scrambled_state: The resulting scrambled state.
    """
    state = env.reset()
    for _ in range(scramble_depth):
        action = env.action_space.sample()
        state, _, _, _ = env.step(action)
    return state


def heuristic_number_of_incorrect_stickers(state):
    """
    Heuristic function to count the number of incorrectly placed stickers.

    Parameters:
    - state: The current state.

    Returns:
    - Number of incorrectly placed stickers.
    """
    incorrect = 0
    for face in state:
        correct_color = face[0, 0]
        incorrect += np.sum(face != correct_color)
    return incorrect


def train_with_systematic_scrambling(agent, env, episodes=1000, scramble_depth=3):
    """
    Train the Q-learning agent with systematic scrambling.

    Parameters:
    - agent: The Q-learning agent.
    - env: The 2x2x2 Rubik's Cube environment.
    - episodes: Number of training episodes.
    - scramble_depth: Number of moves to scramble the cube before each episode.
    """
    for episode in range(episodes):
        state = generate_scramble_sequence(env, scramble_depth)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()


def train_with_guided_exploration(agent, env, episodes=1000):
    """
    Train the Q-learning agent with guided exploration based on state complexity.

    Parameters:
    - agent: The Q-learning agent.
    - env: The 2x2x2 Rubik's Cube environment.
    - episodes: Number of training episodes.
    """
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)

            # Calculate the heuristic to guide exploration
            complexity = heuristic_number_of_incorrect_stickers(next_state)
            print(f"Episode {episode}, State Complexity: {complexity}")
            state = next_state
        agent.decay_epsilon()


def train_with_heuristic_action_selection(agent, env, episodes=1000):
    """
    Train the Q-learning agent using heuristic-based action selection.

    Parameters:
    - agent: The Q-learning agent.
    - env: The 2x2x2 Rubik's Cube environment.
    - episodes: Number of training episodes.
    """
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # Choose an action based on a heuristic function
            if agent.epsilon < 0.2:
                action = agent.choose_action(state)
            else:
                action = heuristic_guided_action_selection(agent, state, env)

            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()


def heuristic_guided_action_selection(agent, state, env):
    """
    Select an action based on heuristic evaluation.

    Parameters:
    - agent: The Q-learning agent.
    - state: The current state.
    - env: The environment.

    Returns:
    - Action index.
    """
    best_action = None
    best_heuristic = float('inf')

    for action in range(env.action_space.n):
        # Apply action temporarily
        env_copy = env  # Assuming environment supports copying
        next_state, _, _, _ = env_copy.step(action)
        heuristic_value = heuristic_number_of_incorrect_stickers(next_state)

        if heuristic_value < best_heuristic:
            best_heuristic = heuristic_value
            best_action = action

    return best_action if best_action is not None else env.action_space.sample()


# Example training loop
if __name__ == "__main__":
    env = TwoByTwoCubeEnv()
    agent = QLearningAgent(env.action_space)

    print("Training with systematic scrambling...")
    train_with_systematic_scrambling(agent, env, episodes=500, scramble_depth=3)

    print("Training with guided exploration...")
    train_with_guided_exploration(agent, env, episodes=500)

    print("Training with heuristic-based action selection...")
    train_with_heuristic_action_selection(agent, env, episodes=500)
