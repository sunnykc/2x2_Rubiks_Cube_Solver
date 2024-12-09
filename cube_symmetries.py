# cube_symmetries.py

import numpy as np

class CubeSymmetries:
    def __init__(self):
        # Define face indices for easier rotation handling
        self.FRONT, self.BACK, self.LEFT, self.RIGHT, self.UP, self.DOWN = range(6)
    
    def rotate_state(self, state, axis, turns=1):
        """
        Rotates the cube state around a given axis a specified number of turns.
        
        Parameters:
        - state (np.ndarray): The current state of the cube.
        - axis (str): The axis to rotate around ('x', 'y', 'z').
        - turns (int): Number of 90-degree turns to rotate (positive for clockwise).
        
        Returns:
        - np.ndarray: The rotated state.
        """
        for _ in range(turns % 4):
            state = self._rotate_once(state, axis)
        return state
    
    def _rotate_once(self, state, axis):
        """
        Performs a single 90-degree rotation around the specified axis.
        
        Parameters:
        - state (np.ndarray): The current state of the cube.
        - axis (str): The axis to rotate around ('x', 'y', 'z').
        
        Returns:
        - np.ndarray: The rotated state.
        """
        new_state = state.copy()
        if axis == 'x':
            # Rotate Front, Up, Back, Down faces
            new_state[self.FRONT] = np.rot90(state[self.UP], -1)
            new_state[self.UP] = state[self.BACK]
            new_state[self.BACK] = np.rot90(state[self.DOWN], 1)
            new_state[self.DOWN] = state[self.FRONT]
            # Rotate Left and Right faces
            new_state[self.LEFT] = np.rot90(state[self.LEFT], -1)
            new_state[self.RIGHT] = np.rot90(state[self.RIGHT], 1)
        elif axis == 'y':
            # Rotate Front, Right, Back, Left faces
            new_state[self.FRONT] = state[self.LEFT]
            new_state[self.LEFT] = state[self.BACK]
            new_state[self.BACK] = state[self.RIGHT]
            new_state[self.RIGHT] = state[self.FRONT]
            # Rotate Up and Down faces
            new_state[self.UP] = np.rot90(state[self.UP], 1)
            new_state[self.DOWN] = np.rot90(state[self.DOWN], -1)
        elif axis == 'z':
            # Rotate Up, Front, Down, Back faces
            new_state[self.UP] = np.rot90(state[self.UP], -1)
            new_state[self.FRONT] = state[self.UP]
            new_state[self.DOWN] = state[self.FRONT]
            new_state[self.BACK] = state[self.DOWN]
            # Rotate Left and Right faces
            new_state[self.LEFT] = np.rot90(state[self.LEFT], 1)
            new_state[self.RIGHT] = np.rot90(state[self.RIGHT], -1)
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
        return new_state
    
    def get_all_symmetries(self, state):
        """
        Generates all symmetrical equivalents of a given state.
        
        Parameters:
        - state (np.ndarray): The current state of the cube.
        
        Returns:
        - list of tuples: All symmetrical state tuples.
        """
        symmetries = []
        axes = ['x', 'y', 'z']
        turns = [0, 1, 2, 3]  # 0 to 270 degrees
        
        for axis in axes:
            for turn in turns:
                rotated_state = self.rotate_state(state, axis, turn)
                symmetries.append(state_to_tuple(rotated_state))
        return symmetries
    
    def get_canonical_state(self, state):
        """
        Determines the canonical state among all symmetrical equivalents.
        
        Parameters:
        - state (np.ndarray): The current state of the cube.
        
        Returns:
        - tuple: The canonical state tuple.
        """
        symmetries = self.get_all_symmetries(state)
        canonical = min(symmetries)
        return canonical

def state_to_tuple(state):
    """
    Convert the state array to a tuple for use as a dictionary key.
    
    Parameters:
    - state (np.ndarray): The cube state as integer indices.
    
    Returns:
    - tuple: A tuple representation of the state.
    """
    return tuple(state.flatten())
