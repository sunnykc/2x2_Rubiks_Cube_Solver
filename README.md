# 2x2x2 Rubik's Cube Solver using Reinforcement Learning (Tabular Q Learning)

Q-Learning Agent for Solving a 2x2x2 Rubik's Cube
-------------------------------------------------

Description:
The 2x2x2 Rubik's Cube is a simpler version of the classic 3x3x3 Rubik's Cube, yet it still presents a significant challenge due to its large number of possible states and rotations. This project implements a Q-learning agent that can learn to solve the 2x2x2 Rubik's Cube by interacting with a simulated environment.

The agent learns by trying random actions at first, observing their effects, and receiving rewards for progress towards the solved cube. Over time, it builds a Q-table that stores the expected rewards for taking specific actions in specific states, enabling it to choose the most efficient moves to solve the cube.

Key reinforcement learning techniques are used to improve the agent's learning process:
- **Epsilon-Greedy Policy**: Balances exploration (trying new actions) and exploitation (using the best-known actions) during training.
- **Symmetry Reduction**: Identifies equivalent cube states under rotations and reflections, reducing the size of the state space and improving efficiency.
- **Replay Buffer**: Stores past experiences and samples them randomly during training to stabilize learning and break correlations between consecutive actions.
- **Reward Function**: Rewards the agent for getting closer to solving the cube, penalizes inefficient moves, and gives a large reward for completely solving it.

The project provides a custom Gym environment for the 2x2x2 Rubik's Cube, allowing the agent to interact with the cube using a predefined action space (rotations of the cube's faces). The training script builds the Q-table, and the demonstration script shows the agent solving scrambled cubes step-by-step using its learned policy.

**Training Time and Process**:
Training the agent to solve the 2x2x2 Rubik's Cube requires running thousands of episodes, with each episode consisting of multiple steps. An episode begins with the cube in a scrambled state and ends when the agent solves the cube or reaches a maximum number of steps. During each step, the agent selects an action, updates its Q-table, and accumulates rewards.

Due to the large state-action space, training is computationally intensive and can take significant time depending on the number of episodes and steps per episode:
- **Longer Training**: Running more episodes allows the agent to explore a wider range of states, leading to better policy convergence.
- **Episode Length**: Allowing more steps per episode gives the agent more time to reach the goal but increases computational time per episode.

The trade-off between training duration and solution quality is important. While shorter training might produce a functional but suboptimal agent, longer training ensures the agent learns more efficient solving strategies. The final Q-table represents the culmination of all training episodes and reflects the agent's learned ability to solve the cube in minimal steps.


Features:
- Simulated 2x2x2 Rubik's Cube environment.
- Tabular Q-learning for action-value mapping.
- Symmetry reduction to minimize redundant states.
- Replay buffer for stable and efficient training.
- Reward function to guide solving efficiently.

Requirements:
- Python 3.6 or higher
- Libraries: gym, numpy, colorama

How to Use:
1. Install required libraries using:
   pip install -r requirements.txt

2. Train the agent:
   Run `python train_q_learning_agent.py` to train the agent. The trained Q-table will be saved to `q_table_trained.pkl`.

3. Demonstrate solving:
   Run `python demonstrate_agent.py` to see the trained agent solve a scrambled cube.

Project Structure:
- rubiks_cube_env.py: Simulated cube environment.
- q_learning_agent.py: Q-learning implementation.
- replay_buffer.py: Experience replay mechanism.
- train_q_learning_agent.py: Training script.
- demonstrate_agent.py: Demonstration script.
- q_table_trained.pkl: Trained Q-table file.
- generate_q_table_parallel.py: Generates Q Table using 6 cores of CPU
- train_agents_parallely.py: train agents parallely using 6 cores of CPU
- test_cube_rotation.py: test whether the rotation mechanism is implemented correctly



Contact:
For questions or contributions, fork the repository or reach out to the project maintainers.
