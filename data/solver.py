import heapq

class Solver:
    class Node:
        def __init__(self, position, g_cost, h_cost, parent=None):
            self.position = position
            self.g_cost = g_cost
            self.h_cost = h_cost
            self.f_cost = g_cost + h_cost
            self.parent = parent

        def __lt__(self, other):
            return self.f_cost < other.f_cost
    
    @staticmethod
    def manhattan_distance(pos1, pos2):
        """Calculate Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def find_start_end(maze):
        """Find the entrance and exit points of the maze."""
        height, width = maze.grid.shape
        start = end = None
        
        # Check top and bottom rows
        for x in range(width):
            if maze.grid[0, x] == 0:
                start = (0, x)
            if maze.grid[height-1, x] == 0:
                end = (height-1, x)
        
        # Check left and right columns if needed
        if start is None or end is None:
            for y in range(height):
                if maze.grid[y, 0] == 0:
                    if start is None:
                        start = (y, 0)
                    else:
                        end = (y, 0)
                if maze.grid[y, width-1] == 0:
                    if start is None:
                        start = (y, width-1)
                    else:
                        end = (y, width-1)
        
        return start, end
    
    @classmethod
    def solve(cls, maze):
        """Solve the maze using A* algorithm."""
        start, end = cls.find_start_end(maze)
        if start is None or end is None:
            return None

        open_set = []
        closed_set = set()
        start_node = cls.Node(start, 0, cls.manhattan_distance(start, end))
        heapq.heappush(open_set, start_node)
        node_dict = {start: start_node}
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while open_set:
            current = heapq.heappop(open_set)
            
            if current.position == end:
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]
            
            closed_set.add(current.position)
            
            for dy, dx in directions:
                new_pos = (current.position[0] + dy, current.position[1] + dx)
                
                if (0 <= new_pos[0] < maze.grid.shape[0] and 
                    0 <= new_pos[1] < maze.grid.shape[1] and 
                    maze.grid[new_pos] == 0 and 
                    new_pos not in closed_set):
                    
                    new_g = current.g_cost + 1
                    new_h = cls.manhattan_distance(new_pos, end)
                    
                    if new_pos not in node_dict or new_g < node_dict[new_pos].g_cost:
                        new_node = cls.Node(new_pos, new_g, new_h, current)
                        node_dict[new_pos] = new_node
                        heapq.heappush(open_set, new_node)
        
        return None

