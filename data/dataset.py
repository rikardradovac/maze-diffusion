
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms




class MazeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe: Pandas dataframe containing maze data
            transform: Optional transform to be applied on images
        """
        self.data = dataframe
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            # Using mean and std for RGB channels
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # One value per RGB channel
                std=[0.5, 0.5, 0.5]    # One value per RGB channel
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load current maze image in RGB
        current_maze = Image.open(self.data.iloc[idx]['current_maze']).convert('RGB')

        # Apply transform - will result in tensor of shape [3, H, W]
        current_maze = self.transform(current_maze)

        # Check if all other values are None
        if pd.isna(self.data.iloc[idx]['action']):
            return {
                'current_maze': current_maze,  # Shape: [3, H, W]
            }

        # Convert action to one-hot encoding
        action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        action = torch.zeros(4)  # One-hot vector for 4 possible actions
        if self.data.iloc[idx]['action'] != "START":
            action[action_map[self.data.iloc[idx]['action']]] = 1

        return {
            'current_maze': current_maze,  # Shape: [3, H, W]
            'action': action,              # Shape: [4]
        }
    
    def __repr__(self):
        return f"MazeDataset(size={len(self.data)}, transform={self.transform})"
    
    
    
class SequenceMazeDataset(Dataset):
    def __init__(self, dataframe, transform=None, max_frames: int = 32):
        """
        Args:
            dataframe: Pandas dataframe containing maze data
            transform: Optional transform to be applied on images
        """
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        self.max_frames = max_frames
        
        def extract_maze_info(path):
            # For paths like "data/mazes/maze123/maze_0.png"
            parts = path.split('/')
            maze_num = None
            frame_num = None
            
            for part in parts:
                # Extract maze folder number (e.g., "maze123" -> 123)
                if part.startswith('maze') and part[4:].isdigit():
                    maze_num = int(part[4:])
                # Extract frame number (e.g., "maze_0.png" -> 0)
                elif part.startswith('maze_') and '.png' in part:
                    frame_num = int(part.split('_')[1].split('.')[0])
            
            return maze_num, frame_num

        # Add frame number and sort within groups
        dataframe['maze_num'], dataframe['frame_num'] = zip(*dataframe['frame'].apply(extract_maze_info))
        
        # Store grouped sequences, ensuring frames are in correct order
        self.sequences = []
        for maze_num, group in dataframe.groupby('maze_num'):
            if maze_num is not None:  # Only process valid maze groups
                sorted_group = group.sort_values('frame_num').reset_index(drop=True)
                self.sequences.append(sorted_group)
            
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx][:self.max_frames]
        sequence_length = len(sequence)
        
        # Load and transform all images in the sequence
        maze_frames = []
        actions = []
        
        for _, row in sequence.iterrows():
            # Load and transform current maze image
            current_maze = Image.open(row['frame']).convert('RGB')
            current_maze = self.transform(current_maze)
            maze_frames.append(current_maze)
            
            # Convert action to one-hot encoding
            action = torch.zeros(4)  # One-hot vector for 4 possible actions
            if row['action'] is not None:  # Skip for last frame
                action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
                action[action_map[row['action']]] = 1
            actions.append(action)
        
        # Get shape of first maze frame for padding
        C, H, W = maze_frames[0].shape
        
        # Pad sequences if needed
        while len(maze_frames) < self.max_frames:
            maze_frames.append(torch.zeros((C, H, W)))
            actions.append(torch.zeros(4))
        
        # Stack all frames and actions
        maze_sequence = torch.stack(maze_frames)      # Shape: [max_frames, C, H, W]
        action_sequence = torch.stack(actions)        # Shape: [max_frames, 4]
        
        return {
            'maze_sequence': maze_sequence,           # Shape: [max_frames, C, H, W]
            'actions': action_sequence,               # Shape: [max_frames, 4]
            'sequence_length': sequence_length        # Original sequence length before padding
        }
    
    def __repr__(self):
        return f"SequenceMazeDataset(num_sequences={len(self.sequences)})"