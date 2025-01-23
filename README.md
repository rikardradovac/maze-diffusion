# Maze Navigation with Diffusion Models

An approach to maze solving using conditional denoising diffusion models. This project demonstrates how modern generative AI can be applied to sequential decision-making tasks by learning to "imagine" paths through mazes frame by frame.

## Overview

This project implements a unique maze-solving approach using diffusion models instead of traditional pathfinding algorithms. The system generates a sequence of frames showing a path through a maze by gradually denoising random patterns into valid frames.

### Key Features

- Frame-by-frame maze path generation using diffusion models
- Conditional generation based on initial maze context
- Smooth visualization of solution paths

## How It Works

The system employs a conditional denoising diffusion model that:
1. Takes two initial maze frames as context
2. Generates subsequent frames through iterative denoising
3. Produces visually coherent and valid maze solutions

The architecture uses a conditioned UNet that considers:
- Previous maze frames as context
- Learned valid movement patterns
- Maze structure constraints

## Training Data

The model is trained on sequences of maze frames showing valid paths:
- Generated using A* algorithm for optimal path finding
- Clear visualization of paths and walls
- Consistent entrance and exit points
- Highlighted solution paths

## Results

The trained model can:
- Generate valid maze solutions from a given starting point
- Produce smooth, visually coherent path sequences
- Bridge traditional pathfinding with modern generative AI

While not always finding the optimal path, the model consistently generates valid solutions, demonstrating the potential of applying diffusion models to structured decision-making tasks.

## Setup and Usage

To train the model, first generate the datasets:

```bash
python data/generate_data.py --maze_size maze_size --num_train_mazes X  --num_test_mazes Y --images_directory path  --train_filename path --test_filename path --solved --num_steps Z
```

Then run the training script (supports multi-gpu distributed training):

```bash
torchrun --nproc_per_node X train.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## License

MIT