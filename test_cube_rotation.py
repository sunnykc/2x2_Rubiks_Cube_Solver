import numpy as np
from rubiks_cube_env import TwoByTwoCubeEnv

# Mapping of actions to their corresponding reverse actions
# Each action is defined as: (face_idx, direction)
# Reverse action is simply the same face with the opposite direction
ACTION_REVERSE_MAP = {
    action: (action // 2, 'counterclockwise' if action % 2 == 0 else 'clockwise')
    for action in range(12)
}

def get_reverse_action(action):
    """
    Returns the reverse action for a given action.
    
    Parameters:
    - action (int): The action index.
    
    Returns:
    - int: The reverse action index.
    """
    face_idx, direction = ACTION_REVERSE_MAP[action]
    return face_idx * 2 + (0 if direction == 'clockwise' else 1)

def are_states_equal(state1, state2):
    """
    Compares two cube states for equality.
    
    Parameters:
    - state1 (np.ndarray): The first cube state.
    - state2 (np.ndarray): The second cube state.
    
    Returns:
    - bool: True if states are identical, False otherwise.
    """
    return np.array_equal(state1, state2)

def print_cube(state):
    """
    Prints the cube state in a matrix format with color codes.
    
    Parameters:
    - state (np.ndarray): The cube state to print.
    """
    # Matrix representation:
    #      UP
    # LEFT FRONT RIGHT
    #      BACK
    #      DOWN
    print("       ", " ".join(state[4][0]))  # UP top row
    print("       ", " ".join(state[4][1]))  # UP bottom row
    print(" ".join(state[2][0]), " ".join(state[0][0]), " ".join(state[3][0]))  # LEFT, FRONT, RIGHT top rows
    print(" ".join(state[2][1]), " ".join(state[0][1]), " ".join(state[3][1]))  # LEFT, FRONT, RIGHT bottom rows
    print("       ", " ".join(state[1][0]))  # BACK top row
    print("       ", " ".join(state[1][1]))  # BACK bottom row
    print("       ", " ".join(state[5][0]))  # DOWN top row
    print("       ", " ".join(state[5][1]))  # DOWN bottom row
    print()

def test_inverse_rotations():
    """
    Tests that applying a rotation followed by its inverse returns the cube to the solved state.
    """
    env = TwoByTwoCubeEnv()
    env._create_solved_cube()  # Initialize to solved state
    initial_state = env.state.copy()
    all_passed = True

    print("=== Inverse Rotation Tests ===\n")
    print("Initial Solved Cube State:")
    print_cube(initial_state)

    for action in range(env.action_space.n):
        face_idx = action // 2
        direction = 'Clockwise' if action % 2 == 0 else 'Counterclockwise'
        reverse_action = get_reverse_action(action)

        print(f"Testing Action {action}: {direction} rotation on face {face_idx}")

        # Apply the rotation
        env._apply_action(action)
        rotated_state = env.state.copy()
        print("State after rotation:")
        print_cube(rotated_state)

        # Apply the reverse rotation
        env._apply_action(reverse_action)
        reverted_state = env.state.copy()
        print(f"State after applying reverse action {reverse_action}:")
        print_cube(reverted_state)

        # Check if reverted state matches the initial state
        if are_states_equal(initial_state, reverted_state):
            print(f"Test Passed for Action {action}\n")
        else:
            print(f"Test Failed for Action {action}\n")
            all_passed = False

    # Summary
    print("=== Inverse Rotation Test Summary ===")
    if all_passed:
        print("All inverse rotation tests passed successfully!\n")
    else:
        print("Some inverse rotation tests failed.\n")

def test_sequential_rotations():
    """
    Tests applying a sequence of rotations and verifies the final state consistency.
    """
    env = TwoByTwoCubeEnv()
    env._create_solved_cube()  # Initialize to solved state
    initial_state = env.state.copy()
    print("=== Sequential Rotation Tests ===\n")
    print("Initial Solved Cube State:")
    print_cube(initial_state)

    # Define a sequence of actions
    # Example: Front clockwise (0), Down counterclockwise (9), Left clockwise (4)
    sequence = [0, 9, 4]
    print(f"Applying rotation sequence: {sequence}\n")

    # Apply the sequence
    for action in sequence:
        env._apply_action(action)
        print(f"After action {action}:")
        print_cube(env.state)

    # Save the state after first sequence
    final_state_first_sequence = env.state.copy()

    # Reset to solved state and apply the same sequence again
    env._create_solved_cube()
    for action in sequence:
        env._apply_action(action)
    final_state_second_sequence = env.state.copy()

    # Check if both final states are equal
    if are_states_equal(final_state_first_sequence, final_state_second_sequence):
        print("Sequential rotation test passed.\n")
    else:
        print("Sequential rotation test failed.\n")

def main():
    """
    Runs all rotation tests.
    """
    test_inverse_rotations()
    test_sequential_rotations()

if __name__ == "__main__":
    main()
