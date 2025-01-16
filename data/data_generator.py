from tqdm import tqdm
import numpy as np
from solver import Solver
from maze import Maze
import os
import pandas as pd


class DataGenerator:
    def __init__(self, maze: Maze, solver: Solver):
        self.maze = maze
        self.solver = solver
        
        self.actions = {
            (0, 1): "DOWN",
            (0, -1): "UP", 
            (1, 0): "RIGHT",
            (-1, 0): "LEFT"
        }
    
    
    def generate_mazes(self, output_directory: str, num_data_iterations: int = 1, solve_maze: bool = True, resolution: tuple = (64, 64), random_walk=False, num_steps=None):
        """Generates training data consisting of mazes with solution paths.
        
        Args:
            num_data_iterations: Number of maze iterations to generate data for
            
        Returns:
            DataFrame containing maze states, positions and actions
        """

        df = pd.DataFrame(columns=['current_maze', 'action'])
        
        for current_iteration in tqdm(range(num_data_iterations), desc="Generating mazes"):
            # Generate new maze and solution for each iteration
            self.maze.generate()
            
            os.makedirs(f"{output_directory}", exist_ok=True)

            if solve_maze:
                os.makedirs(f"{output_directory}/maze{current_iteration}", exist_ok=True)
                path = self.solver.solve(self.maze)
                
                # Process each step in the solution path
                for path_iter in range(len(path)):
                    # Calculate action that led to this position
                    if path_iter == 0:
                        action = "START"
                    else:
                        current_pos = path[path_iter]
                        prev_pos = path[path_iter - 1]
                        # Calculate action based on how we got to current_pos from prev_pos
                        action = (current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0])
                        action = self.actions[action]
                        
                    # Save current state
                    current_filename = f"{output_directory}/maze{current_iteration}/maze_{path_iter}.png"
                    self.maze.save_solved_maze_image(path[:path_iter+1],
                                                   filename=current_filename,
                                                   target_size=resolution)
                    
                    # Add row to dataframe
                    new_row = pd.DataFrame({
                        'frame': [current_filename],
                        'action': [action]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
            elif random_walk:
                os.makedirs(f"{output_directory}/maze{current_iteration}", exist_ok=True)
                path = self._generate_random_walk(num_steps)
                
                # Create lists to store all data
                all_rows = []
                
                # Process each step in the random walk path
                for path_iter in range(len(path)):
                    current_filename = f"{output_directory}/maze{current_iteration}/maze_{path_iter}.png"
                    self.maze.save_solved_maze_image(path[:path_iter+1],
                                                   filename=current_filename,
                                                   target_size=resolution)
                    
                    # Calculate the action for this frame
                    if path_iter == len(path) - 1:
                        action = None  # Last frame has no next action
                    else:
                        current_pos = path[path_iter]
                        next_pos = path[path_iter + 1]
                        diff = (next_pos[1] - current_pos[1], next_pos[0] - current_pos[0])
                        if diff not in self.actions:
                            raise ValueError(f"Invalid move detected from {current_pos} to {next_pos}. Diff: {diff}")
                        action = self.actions[diff]
                        
                    # Add to our list of rows
                    all_rows.append({
                        'frame': current_filename,  # Current frame
                        'action': action           # Action for this frame
                    })
                
                # Create DataFrame
                if current_iteration == 0:
                    df = pd.DataFrame(all_rows)
                else:
                    df = pd.concat([df, pd.DataFrame(all_rows)], ignore_index=True)
            else:
                current_filename = f"{output_directory}/maze_{current_iteration}.png"
                self.maze.save_maze_image(current_filename, target_size=resolution)
                new_row = pd.DataFrame({
                    'current_maze': [current_filename],
                    'action': [None]
                })
                
                df = pd.concat([df, new_row], ignore_index=True)
                
        return df

    def _generate_random_walk(self, num_steps):
        """Generate a random walk through the maze, prioritizing unvisited adjacent cells."""
        # Find entrance
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if self.maze.grid[y, x] == 0:
                    start = (y, x)
                    break
            if 'start' in locals():
                break

        path = [start]
        current = start
        visited = {start}  # Keep track of visited positions
        steps_taken = 0
        
        while steps_taken < num_steps:
            # Separate possible moves into unvisited and visited
            unvisited_moves = []
            visited_moves = []
            
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
                new_y, new_x = current[0] + dy, current[1] + dx
                if (0 <= new_y < self.maze.height and 
                    0 <= new_x < self.maze.width and 
                    self.maze.grid[new_y, new_x] == 0):
                    
                    new_pos = (new_y, new_x)
                    if new_pos not in visited:
                        unvisited_moves.append(new_pos)
                    else:
                        visited_moves.append(new_pos)
            
            # Prioritize unvisited moves, only use visited moves if necessary
            if unvisited_moves:
                current = unvisited_moves[np.random.randint(len(unvisited_moves))]
            elif visited_moves:
                current = visited_moves[np.random.randint(len(visited_moves))]
            else:
                # If no valid moves at all, stay in place
                current = path[-1]
            
            path.append(current)
            visited.add(current)
            steps_taken += 1
        
        return path[:num_steps]