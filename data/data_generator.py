from tqdm import tqdm
import numpy as np
from solver import Solver
from maze import Maze
import os
import pandas as pd
import multiprocessing as mp
from functools import partial


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
    
    
    @staticmethod
    def _process_single_maze(iteration, maze, solver, output_directory, solve_maze, resolution, random_walk, num_steps, actions):
        """Process a single maze iteration - this will run in parallel"""
        # Create a new maze instance for this process
        maze.generate()
        
        if solve_maze:
            os.makedirs(f"{output_directory}/maze{iteration}", exist_ok=True)
            path = solver.solve(maze)
            
            rows = []
            for path_iter in range(len(path)):
                if path_iter == 0:
                    action = "START"
                else:
                    current_pos = path[path_iter]
                    prev_pos = path[path_iter - 1]
                    action = (current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0])
                    action = actions[action]
                    
                current_filename = f"{output_directory}/maze{iteration}/maze_{path_iter}.png"
                maze.save_solved_maze_image(path[:path_iter+1],
                                       filename=current_filename,
                                       target_size=resolution)
                
                rows.append({
                    'frame': current_filename,
                    'action': action
                })
            return rows
            
        elif random_walk:
            os.makedirs(f"{output_directory}/maze{iteration}", exist_ok=True)
            path = DataGenerator._generate_random_walk(maze, num_steps)
            
            rows = []
            for path_iter in range(len(path)):
                current_filename = f"{output_directory}/maze{iteration}/maze_{path_iter}.png"
                maze.save_solved_maze_image(path[:path_iter+1],
                                       filename=current_filename,
                                       target_size=resolution)
                
                if path_iter == len(path) - 1:
                    action = None
                else:
                    current_pos = path[path_iter]
                    next_pos = path[path_iter + 1]
                    diff = (next_pos[1] - current_pos[1], next_pos[0] - current_pos[0])
                    action = actions[diff]
                
                rows.append({
                    'frame': current_filename,
                    'action': action
                })
            return rows
        
        else:
            current_filename = f"{output_directory}/maze_{iteration}.png"
            maze.save_maze_image(current_filename, target_size=resolution)
            return [{
                'current_maze': current_filename,
                'action': None
            }]

    def generate_mazes(self, output_directory: str, num_data_iterations: int = 1, 
                      solve_maze: bool = True, resolution: tuple = (64, 64), 
                      random_walk=False, num_steps=None):
        """Generates training data consisting of mazes with solution paths in parallel."""
        
        os.makedirs(output_directory, exist_ok=True)
        
        # Determine number of processes to use (leave one core free)
        num_processes = max(1, mp.cpu_count() - 1)
        
        # Create partial function with fixed arguments
        process_maze = partial(
            self._process_single_maze,
            maze=self.maze,
            solver=self.solver,
            output_directory=output_directory,
            solve_maze=solve_maze,
            resolution=resolution,
            random_walk=random_walk,
            num_steps=num_steps,
            actions=self.actions
        )
        
        # Create pool and process mazes in parallel
        with mp.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_maze, range(num_data_iterations)),
                total=num_data_iterations,
                desc="Generating mazes"
            ))
        
        # Flatten results and create DataFrame
        all_rows = [row for result in results for row in result]
        df = pd.DataFrame(all_rows)
        
        return df

    @staticmethod
    def _generate_random_walk(maze, num_steps):
        """Generate a random walk through the maze, prioritizing unvisited adjacent cells."""
        # Find entrance
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.grid[y, x] == 0:
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
                if (0 <= new_y < maze.height and 
                    0 <= new_x < maze.width and 
                    maze.grid[new_y, new_x] == 0):
                    
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