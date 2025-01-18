import numpy as np
from random import choice, randint
from PIL import Image

class Maze:
    def __init__(self, width, height):
        
        if width % 2 == 0: width += 1
        if height % 2 == 0: height += 1
        self.width = width
        self.height = height
        self.grid = np.ones((height, width), dtype=np.uint8)


    def generate(self):
        """Generate a new maze with given dimensions."""
        # Set a new random seed for each maze generation
        np.random.seed(None)  # Reset numpy's random seed
        
        # Ensure odd dimensions
        self.grid = np.ones((self.height, self.width), dtype=np.uint8)
        self._carve_path(1, 1)
        self._create_random_openings()
    
    def _create_random_openings(self):
        """Create entrance and exit on random but opposite sides of the maze."""
        sides = ['top', 'bottom', 'left', 'right']
        entrance_side = choice(sides)
        
        opposite_sides = {
            'top': 'bottom',
            'bottom': 'top',
            'left': 'right',
            'right': 'left'
        }
        exit_side = opposite_sides[entrance_side]
        
        self._create_opening(entrance_side)
        self._create_opening(exit_side)
    
    def _create_opening(self, side):
        """Create an opening on the specified side."""
        if side == 'top':
            for attempt in range(100):
                x = randint(1, self.width-2)
                if self.grid[1, x] == 0:
                    self.grid[0, x] = 0
                    return
                    
        elif side == 'bottom':
            for attempt in range(100):
                x = randint(1, self.width-2)
                if self.grid[self.height-2, x] == 0:
                    self.grid[self.height-1, x] = 0
                    return
                    
        elif side == 'left':
            for attempt in range(100):
                y = randint(1, self.height-2)
                if self.grid[y, 1] == 0:
                    self.grid[y, 0] = 0
                    return
                    
        elif side == 'right':
            for attempt in range(100):
                y = randint(1, self.height-2)
                if self.grid[y, self.width-2] == 0:
                    self.grid[y, self.width-1] = 0
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
        """Save the maze with the solution path highlighted in red."""
        img_array = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        img_array[self.grid == 1] = [0, 0, 0]      # walls = black
        img_array[self.grid == 0] = [255, 255, 255] # paths = white
        
        if path:
            for y, x in path:
                img_array[y, x] = [255, 0, 0]  # solution = red
        
        img = Image.fromarray(img_array)
        img_resized = img.resize(target_size, Image.NEAREST)
        img_resized.save(filename)
        return img_resized

    def print_maze(self):
        """Print ASCII representation of the maze."""
        for row in self.grid:
            print(''.join(['█' if cell == 1 else ' ' for cell in row]))

    def create_maze_frame(self, highlight_cells=None, target_size=(32, 32), entrances=None):
        """Create a maze frame with specified cells highlighted in red, returning the image buffer."""
        # Create initial array
        img_array = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        img_array[self.grid == 1] = [0, 0, 0]      # walls = black
        img_array[self.grid == 0] = [255, 255, 255] # paths = white
        
        # Make entrance positions white
        if entrances:
            for entrance in entrances:
                y, x = entrance.y, entrance.x
                img_array[y, x] = [255, 255, 255]  # entrances = white
        
        # Highlight specified cells in red
        if highlight_cells:
            for cell in highlight_cells:
                y, x = cell['y'], cell['x']
                img_array[y, x] = [255, 0, 0]  # highlight = red
        
        # Create and resize image
        img = Image.fromarray(img_array)
        img_resized = img.resize(target_size, Image.NEAREST)
        
        return img_resized
