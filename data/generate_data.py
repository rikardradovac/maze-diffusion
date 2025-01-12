import argparse
from maze import Maze
from solver import Solver
from data_generator import DataGenerator
import os

def generate_unsolved_mazes(maze_size, num_mazes, resolution, train_ratio, images_directory, train_filename, test_filename, random_seed=42):
    maze = Maze(maze_size, maze_size)
    solver = Solver()
    dg = DataGenerator(maze, solver)
    
    os.makedirs(os.path.dirname(train_filename), exist_ok=True)

    

    df = dg.generate_mazes(images_directory, num_mazes, resolution=resolution, solve_maze=False)

    train_size = int(train_ratio * len(df))
    train_df = df.sample(n=train_size, random_state=random_seed)
    test_df = df.drop(train_df.index)

    train_df.to_parquet(train_filename)
    test_df.to_parquet(test_filename)

def generate_solved_mazes(maze_size, num_train_mazes, num_test_mazes, resolution, images_directory, train_filename, test_filename):
    maze = Maze(maze_size, maze_size)
    solver = Solver()
    dg = DataGenerator(maze, solver)
    
    os.makedirs(os.path.dirname(train_filename), exist_ok=True)

    # Generate training data with solved paths
    train_df = dg.generate_mazes(images_directory + "_train", num_train_mazes, resolution=resolution, solve_maze=True)
    train_df.to_parquet(train_filename)

    # Generate separate test data with solved paths
    test_df = dg.generate_mazes(images_directory + "_test", num_test_mazes, resolution=resolution, solve_maze=True)
    test_df.to_parquet(test_filename)

def generate_random_walk_mazes(maze_size, num_mazes, num_steps, resolution, images_directory, train_filename, test_filename, random_seed=42):
    maze = Maze(maze_size, maze_size)
    solver = Solver()
    dg = DataGenerator(maze, solver)
    
    os.makedirs(os.path.dirname(train_filename), exist_ok=True)

    # Generate training data
    train_df = dg.generate_mazes(images_directory + "_train", num_mazes, resolution=resolution, 
                                solve_maze=False,
                                random_walk=True, 
                                num_steps=num_steps)
    train_df.to_parquet(train_filename)

    # Generate separate test data
    test_df = dg.generate_mazes(images_directory + "_test", int(0.2 * num_mazes), resolution=resolution,
                               solve_maze=False,
                               random_walk=True,
                               num_steps=num_steps)
    test_df.to_parquet(test_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mazes and save to parquet files.")
    parser.add_argument("--maze_size", type=int, default=16, help="Size of the maze (default: 16)")
    parser.add_argument("--num_train_mazes", type=int, default=500, help="Number of training mazes to generate (default: 500)")
    parser.add_argument("--num_test_mazes", type=int, default=100, help="Number of test mazes to generate (default: 100)")
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the mazes (default: 32)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data for unsolved mazes (default: 0.8)")
    parser.add_argument("--images_directory", type=str, default="data/mazes", help="Base output directory for the generated mazes")
    parser.add_argument("--train_filename", type=str, default="data/vit_data/train_dataset.parquet", help="Output filename for training dataset")
    parser.add_argument("--test_filename", type=str, default="data/vit_data/test_dataset.parquet", help="Output filename for testing dataset")
    parser.add_argument("--solved", action="store_true", help="Generate solved mazes if set")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for dataset splitting (default: 42)")
    parser.add_argument("--random_walk", action="store_true", help="Generate mazes with random walks if set")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps for random walk (default: 100)")

    args = parser.parse_args()
    
    resolution = (args.resolution, args.resolution)
    if args.solved:
        generate_solved_mazes(args.maze_size, args.num_train_mazes, args.num_test_mazes, 
                            resolution, args.images_directory, args.train_filename, args.test_filename)
    elif args.random_walk:
        generate_random_walk_mazes(args.maze_size, args.num_train_mazes, args.num_steps,
                                 resolution, args.images_directory, args.train_filename, 
                                 args.test_filename, args.random_seed)
    else:
        generate_unsolved_mazes(args.maze_size, args.num_train_mazes, resolution, args.train_ratio,
                              args.images_directory, args.train_filename, 
                              args.test_filename, args.random_seed)