from collections import deque
import sys


def num_islands(map_grid):
    """
    Main function that calculates number of islands with the help of two nested helper functions.  

    Args:
        map_grid: A 2d matrix(list of lists) containing topography information of  "land" and "water". 
                Land is denoted by char "1" water is denoted by char "0".

    Returns:
        num_islands: Number of separate islands in the map
    """
    rows, cols = len(map_grid), len(map_grid[0])

    visited = set()

    def is_valid(i, j):
        """Given i,j as matrix co-ordiantes, Checks if-
        co-ordinate is valid and 
        co-ordinate is land (denoted by 1) and
        co-ordinate in unexplored ( not in visited set)

        Args:
            i: row index
            j: column index

        Returns:
            True or False 
        """
        # nonlocal to use variables from enclosing method
        nonlocal rows, cols
        if i > -1 and i < rows and j > -1 and j < cols and map_grid[i][j] == "1" and (i, j) not in visited:
            return True
        else:
            return False

    def bfs(source):
        """Perform Breadth first exploration of valid land area of the 2-d map. 

        Args:
            source: tuple (i,j) where i is row index and j is column index
        Returns:
            None
        """
        nonlocal visited

        # Only explore horizontally or vertical directions (no diagonal exploration).
        directions = [[0, 1], [0, -1], [-1, 0], [1, 0]]

        q = deque([source])
        # print(source)
        while (q):
            # print(q)
            cur_i, cur_j = q.popleft()

            for next_i, next_j in directions:
                dir_i, dir_j = cur_i+next_i, cur_j+next_j
                if (is_valid(dir_i, dir_j)):
                    q.append((dir_i, dir_j))
                    visited.add((dir_i, dir_j))

        return

    # Iterate through non-contiguous chunks of "land"
    num_islands = 0
    for i in range(rows):
        for j in range(cols):
            # print('test')
            if map_grid[i][j] == "1" and (i, j) not in visited:
                # print(i,j)
                bfs((i, j))
                num_islands += 1

    return num_islands


def test_solution():
    """optional helper function to test implementation
    """
    case_1 = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"],
              ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]
    case_2 = [["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"],
              ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]]
    assert num_islands(case_1) == 1
    assert num_islands(case_2) == 3


def get_inputs():
    """Reads data from stdin, parses and returns 2d matrix containing char "0" or "1". 

    Returns:
        grid_map: list of lists
    """
    grid_map = []

    while True:
        row = list(map(str, sys.stdin.readline().strip().split()))
        if len(row) > 0:
            grid_map.append(row)
        else:
            break
    # print(grid_map)
    return grid_map


# test_solution()
print(num_islands(get_inputs()))
