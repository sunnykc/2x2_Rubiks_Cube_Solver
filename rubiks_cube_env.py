# rubiks_cube_env.py

import gym
import numpy as np
from gym import spaces

class TwoByTwoCubeEnv(gym.Env):
    """
    A simplified 2x2x2 cube environment for reinforcement learning.
    Each face is a 2x2 matrix of characters representing colors (W, Y, B, G, R, O).
    Goal: All stickers on each face should match the face's center color.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TwoByTwoCubeEnv, self).__init__()
        # Define color codes for the faces
        # Order: 0=Front, 1=Back, 2=Left, 3=Right, 4=Up, 5=Down
        self.colors = ['W', 'Y', 'B', 'G', 'R', 'O']  # White, Yellow, Blue, Green, Red, Orange

        # State: 6 faces, each 2x2
        self.state = np.empty((6, 2, 2), dtype='<U1')  # Unicode string of length 1 for color codes

        # Action space: 6 faces * 2 directions = 12 possible moves
        # Even actions = clockwise, Odd actions = counterclockwise
        self.action_space = spaces.Discrete(12)

        # Observation space: Using MultiDiscrete to represent color codes as integers
        self.observation_space = spaces.MultiDiscrete([6] * 24)  # 6 possible colors for each of the 24 stickers

        self._create_solved_cube()

        # Adjacency mapping for each face rotation
        self.adjacent_map = {
            # FRONT (0) rotation cycles edges from U bottom row -> L right col -> D top row -> R left col
            0: [(4, 1, False), (2, 1, True), (5, 0, False), (3, 0, True)],
            # BACK (1) rotation cycles edges from U top row -> R right col -> D bottom row -> L left col
            1: [(4, 0, False), (3, 1, True), (5, 1, False), (2, 0, True)],
            # LEFT (2) rotation cycles edges from U left col -> F left col -> D left col -> B right col
            2: [(4, 0, True), (0, 0, True), (5, 0, True), (1, 1, True)],
            # RIGHT (3) rotation cycles edges from U right col -> B left col -> D right col -> F right col
            3: [(4, 1, True), (1, 0, True), (5, 1, True), (0, 1, True)],
            # UP (4) rotation cycles edges from B top row -> R top row -> F top row -> L top row
            4: [(1, 0, False), (3, 0, False), (0, 0, False), (2, 0, False)],
            # DOWN (5) rotation cycles edges from F bottom row -> R bottom row -> B bottom row -> L bottom row
            5: [(0, 1, False), (3, 1, False), (1, 1, False), (2, 1, False)],
        }

    def _create_solved_cube(self):
        """Initialize the cube to the solved state."""
        for i, color in enumerate(self.colors):
            self.state[i, :, :] = color  # Each face is uniform in color

    def reset(self):
        """Reset the cube to a scrambled state."""
        self._create_solved_cube()
        # Apply random moves to scramble the cube
        for _ in range(10):  # Scramble with 10 random moves
            action = self.action_space.sample()
            self._apply_action(action)
        # Convert state to integers for observation
        state_int = np.vectorize(self.colors.index)(self.state)
        return state_int.copy()

    def _apply_action(self, action):
        """Apply the given action (face rotation)."""
        face_idx = action // 2
        direction = 'clockwise' if action % 2 == 0 else 'counterclockwise'
        self._rotate_face(face_idx, direction)

    def _rotate_face(self, face_idx, direction):
        """Rotate the specified face and update adjacent faces."""
        # Rotate the stickers on the face itself
        if direction == 'clockwise':
            self.state[face_idx] = np.rot90(self.state[face_idx], -1)
        else:
            self.state[face_idx] = np.rot90(self.state[face_idx], 1)

        # Update adjacent faces
        self._update_adjacent_faces(face_idx, direction)

    def _update_adjacent_faces(self, face_idx, direction):
        """Update rows/columns of adjacent faces when a face is rotated."""
        adj = self.adjacent_map[face_idx]

        # Extract edges in defined order
        edges_before = []
        for face, idx, is_col in adj:
            if is_col:
                edges_before.append(tuple(self.state[face, :, idx]))
            else:
                edges_before.append(tuple(self.state[face, idx, :]))

        # Rotate edges based on direction
        if direction == 'clockwise':
            edges_after = [edges_before[-1]] + edges_before[:-1]
        elif direction == 'counterclockwise':
            edges_after = edges_before[1:] + [edges_before[0]]
        else:
            raise ValueError("Invalid rotation direction. Choose 'clockwise' or 'counterclockwise'.")

        # Write rotated edges back to adjacent faces
        for (face, idx, is_col), edge in zip(adj, edges_after):
            if is_col:
                self.state[face, :, idx] = edge  # Write column
            else:
                self.state[face, idx, :] = edge  # Write row

    def step(self, action):
        """Perform the action and return the new state, reward, done, and info."""
        correct_before = self._count_correct_stickers()
        self._apply_action(action)
        correct_after = self._count_correct_stickers()

        # Reward based on improvement
        reward = (correct_after - correct_before) - 0.1  # Step cost
        done = self._is_solved()

        # Add a large reward if solved
        if done:
            reward += 100

        # Convert state to integers for observation
        state_int = np.vectorize(self.colors.index)(self.state)

        return state_int.copy(), reward, done, {}

    def _is_solved(self):
        """Check if the cube is solved."""
        for face in range(6):
            if not np.all(self.state[face] == self.state[face, 0, 0]):
                return False
        return True

    def _count_correct_stickers(self):
        """Count the number of stickers that match their face's center color."""
        count = 0
        for face in range(6):
            center_color = self.state[face, 0, 0]
            count += np.sum(self.state[face] == center_color)
        return count

    def set_state(self, state_int):
        """
        Set the cube's state directly.
        
        Parameters:
        - state_int (np.ndarray): The cube state as integer indices.
        """
        # Convert integer state to color characters
        state_str = np.vectorize(lambda x: self.colors[x])(state_int)
        self.state = state_str.copy()

    def render(self, mode='human'):
        """Display the cube in a matrix format."""
        # Matrix representation:
        #      UP
        # LEFT FRONT RIGHT
        #      BACK
        #      DOWN
        print("       ", " ".join(self.state[4][0]))  # UP top row
        print("       ", " ".join(self.state[4][1]))  # UP bottom row
        print(" ".join(self.state[2][0]), " ".join(self.state[0][0]), " ".join(self.state[3][0]))  # LEFT, FRONT, RIGHT top rows
        print(" ".join(self.state[2][1]), " ".join(self.state[0][1]), " ".join(self.state[3][1]))  # LEFT, FRONT, RIGHT bottom rows
        print("       ", " ".join(self.state[1][0]))  # BACK top row
        print("       ", " ".join(self.state[1][1]))  # BACK bottom row
        print("       ", " ".join(self.state[5][0]))  # DOWN top row
        print("       ", " ".join(self.state[5][1]))  # DOWN bottom row
        print()
