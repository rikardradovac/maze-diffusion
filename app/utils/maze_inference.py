import numpy as np
from random import choice, randint, sample
from PIL import Image

class Maze:
    def __init__(self, width, height):
        
        if width % 2 == 0: width += 1
        if height % 2 == 0: height += 1
        self.width = width
        self.height = height
        self.grid = np.ones((height, width), dtype=np.uint8)
        self.entrance = None
        self.exit = None


    def generate(self, random_start=True):
        """Generate a new maze with given dimensions.
        
        Parameters: 
            random_start (bool): If True, start carving from a random odd cell.
                                 Otherwise, start at (1, 1).
        """
        # Set a new random seed for each maze generation
        np.random.seed(None)  # Reset numpy's random seed
        
        # Ensure odd dimensions
        self.grid = np.ones((self.height, self.width), dtype=np.uint8)
        
        if random_start:
            # Choose a random starting cell among odd indices (1,3,5,...)
            possible_x = list(range(1, self.width, 2))
            possible_y = list(range(1, self.height, 2))
            start_x = choice(possible_x)
            start_y = choice(possible_y)
        else:
            start_x, start_y = 1, 1
        
        self._carve_path(start_x, start_y)
        self._create_random_openings()
    
    def _create_random_openings(self):
        """Create two openings (entrance and exit) on two random sides of the maze.
        
        This method now selects two distinct sides from ['top', 'bottom', 'left', 'right'],
        meaning that the entrance/exit can be either opposite or adjacent.
        """
        sides = ['top', 'bottom', 'left', 'right']
        entrance_side, exit_side = sample(sides, 2)  # randomly choose two distinct sides

        self._create_opening(entrance_side)
        self._create_opening(exit_side)
    
    def _create_opening(self, side):
        """Create an opening on the specified side."""
        if side == 'top':
            for attempt in range(100):
                x = randint(1, self.width-2)
                if self.grid[1, x] == 0:
                    self.grid[0, x] = 0
                    self.entrance = (0, x)
                    return
                    
        elif side == 'bottom':
            for attempt in range(100):
                x = randint(1, self.width-2)
                if self.grid[self.height-2, x] == 0:
                    self.grid[self.height-1, x] = 0
                    self.exit = (self.height-1, x)
                    return
                    
        elif side == 'left':
            for attempt in range(100):
                y = randint(1, self.height-2)
                if self.grid[y, 1] == 0:
                    self.grid[y, 0] = 0
                    self.entrance = (y, 0)
                    return
                    
        elif side == 'right':
            for attempt in range(100):
                y = randint(1, self.height-2)
                if self.grid[y, self.width-2] == 0:
                    self.grid[y, self.width-1] = 0
                    self.exit = (y, self.width-1)
                    return
    
    def _carve_path(self, x, y):
        """Recursively carve paths through the maze."""
        self.grid[y, x] = 0
        
        directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        np.random.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if (0 < new_x < self.width-1 and 
                0 < new_y < self.height-1 and 
                self.grid[new_y, new_x] == 1):
                
                self.grid[y + dy//2, x + dx//2] = 0
                self._carve_path(new_x, new_y)

    
    def save_maze_image(self, filename='maze.png', target_size=(64, 64)):
        """Save the maze as a PNG image."""
        img_array = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        img_array[self.grid == 1] = [0, 0, 0]      # walls = black
        img_array[self.grid == 0] = [255, 255, 255] # paths = white
        
        img = Image.fromarray(img_array)
        img_resized = img.resize(target_size, Image.NEAREST)
        img_resized.save(filename)
        return img_resized

    def save_solved_maze_image(self, path, filename='solved_maze.png', target_size=(64, 64)):
        """Save the maze with the solution path highlighted in red and exit in green.
        The current position (last point in path) will be shown in blue."""
        img_array = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        img_array[self.grid == 1] = [0, 0, 0]      # walls = black
        img_array[self.grid == 0] = [255, 255, 255] # paths = white
        
        # Mark exit in green (only if it's not part of the path)
        if self.exit and (path is None or self.exit not in path):
            img_array[self.exit[0], self.exit[1]] = [0, 255, 0]  # exit = green
        
        # Path takes priority over exit marking
        if path:
            # Color all points in the path except the last one red
            for y, x in path[:-1]:
                img_array[y, x] = [255, 0, 0]  # path = red
            
            # Color the current position (last point) blue
            if len(path) > 0:
                y, x = path[-1]
                img_array[y, x] = [0, 0, 255]  # current = blue
        
        img = Image.fromarray(img_array)
        img_resized = img.resize(target_size, Image.NEAREST)
        img_resized.save(filename)
        return img_resized

    def print_maze(self):
        """Print ASCII representation of the maze."""
        for row in self.grid:
            print(''.join(['█' if cell == 1 else ' ' for cell in row]))

    def create_maze_frame(self, target_size=(64, 64), start_cell=None, end_cell=None, first_move_cell=None):
        """Create two maze frames:
        1. First frame shows start (red) and end (green)
        2. Second frame shows start (red), end (green), and first move (blue)
        
        Returns: List of two PIL Images
        """
        # Create base maze structure (repeated for both frames)
        base_array = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        base_array[self.grid == 1] = [0, 0, 0]        # walls = black
        base_array[self.grid == 0] = [255, 255, 255]  # paths = white
        
        # Create two identical arrays for our two frames
        frame1 = base_array.copy()
        frame2 = base_array.copy()
        
        # Add start (red) and end (green) points to both frames
        if start_cell:
            frame1[start_cell['y'], start_cell['x']] = [255, 0, 0]  # red
            frame2[start_cell['y'], start_cell['x']] = [255, 0, 0]  # red
        
        if end_cell:
            frame1[end_cell['y'], end_cell['x']] = [0, 255, 0]  # green
            frame2[end_cell['y'], end_cell['x']] = [0, 255, 0]  # green
        
        # Add first move (blue) only to second frame
        if first_move_cell:
            frame2[first_move_cell['y'], first_move_cell['x']] = [0, 0, 255]  # blue
        
        # Convert to PIL Images and resize
        img1 = Image.fromarray(frame1).resize(target_size, Image.NEAREST)
        img2 = Image.fromarray(frame2).resize(target_size, Image.NEAREST)
        
        return [img1, img2]