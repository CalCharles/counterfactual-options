import heapq
from file_management import printframe
import numpy as np

def visualize_path(obstacle_indices, start_grid, goal_grid, path_indices):
    grid = np.zeros(shape=(5,5,3))
    # print(obstacle_indices, path_indices)
    obstacle_indices = np.array(list(obstacle_indices))
    path_indices = np.array(path_indices).astype(int)
    # print(obstacle_indices)
    start_grid = start_grid.astype(int)
    goal_grid = goal_grid.astype(int)
    for os in obstacle_indices:
        grid[os[0], os[1],0] = 1
    for ps in path_indices:
        grid[ps[0], ps[1], 1] = 1
    grid[start_grid[0], start_grid[1], 2] = 1
    grid[goal_grid[0], goal_grid[1],2] = 1
    # print(grid)
    for i in range(30):
        printframe(grid, waittime=20)


# env.table_offset, env.SPAWN_AREA_SIZE, env.OBSTACLE_HALF_SIDELENGTH
def find_path(obstacles, table_offset, spawn_size, obstacle_half_sidelength, grid_resolution, start_position, goal_position):
    obstacle_grid_index = (
            (obstacles[:, :2] - table_offset[:2] + spawn_size / 2) // (obstacle_half_sidelength * 2)
    ).astype(int)

    grid_pos_ref = dict()
    # print(spawn_size, (obstacle_half_sidelength * 2))
    for i in range(int(spawn_size // (obstacle_half_sidelength * 2))):
        for j in range(int(spawn_size // (obstacle_half_sidelength * 2))):
            offset = np.array([obstacle_half_sidelength * 2 * i + obstacle_half_sidelength, obstacle_half_sidelength * 2 * j + obstacle_half_sidelength])
            grid_pos_ref[(i,j)] = table_offset[:2] - spawn_size / 2 + offset
            grid_pos_ref[(i,j)] = np.array(grid_pos_ref[(i,j)].tolist() + [0.0])
    start_grid = ((start_position[:2] - table_offset[:2] + spawn_size / 2) // (obstacle_half_sidelength * 2))
    goal_grid = ((goal_position[:2] - table_offset[:2] + spawn_size / 2) // (obstacle_half_sidelength * 2))
    # print(start_position, goal_position, start_grid, goal_grid, table_offset[:2], spawn_size, (start_position[:2] - table_offset[:2] + spawn_size / 2), grid_pos_ref)

    if tuple(start_grid) == tuple(goal_grid):
        return True, [start_position, goal_position]

    def recover_path(from_ptrs, end_grid, end_pos, start_pos):
        at_grid = tuple(end_grid)
        
        id_path = [at_grid]
        path = [end_pos]
        while at_grid is not None:
            # print(at_grid)
            at_grid = tuple(at_grid)
            at_pos = grid_pos_ref[at_grid]
            path.append(at_pos)
            at_grid = from_ptrs[at_grid]
            id_path.append(at_grid)
        id_path.pop(-1)
        path.append(start_pos)
        path.reverse()
        id_path.reverse()
        return path, id_path


    obstacle_indices = set(map(tuple, obstacle_grid_index))
    queue = [(np.linalg.norm(start_grid - goal_grid), 0, tuple(start_grid))]  # each element is (total_cost, forward_cost, position)
    from_ptrs = {tuple(start_grid): None}
    visited = set()
    print(start_grid, goal_grid, start_position, goal_position, obstacle_grid_index, obstacles)
    visualize_path(obstacle_indices, start_grid, goal_grid, [])
    while queue:
        _, fcost, pos = heapq.heappop(queue)
        if pos in visited:
            continue
        # check if goal circle intersects the current grid square
        # closest = np.maximum(pos, np.minimum(np.array(pos) + 1, goal_pos))
        # if np.linalg.norm(closest - goal_pos) <= self.GOAL_RADIUS:
        # print(pos, goal_grid)
        if pos == tuple(goal_grid):
            path, id_path =recover_path(from_ptrs, goal_grid, goal_position, start_position)
            visualize_path(obstacle_indices, start_grid, goal_grid, id_path)
            return True, path 
        visited.add(pos)
        for direction in np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]):
            next_pos = pos + direction
            if (
                np.any(next_pos < 0) or
                np.any(next_pos >= grid_resolution) or
                tuple(next_pos) in obstacle_indices
            ):
                continue
            if tuple(next_pos) not in from_ptrs:
                from_ptrs[tuple(next_pos)] = pos # this line is safe because all edge costs are the same
            print(next_pos, from_ptrs[tuple(next_pos)], pos)
            heapq.heappush(queue, (np.linalg.norm(next_pos - goal_grid) + fcost + 1, fcost + 1, tuple(next_pos)))
    return False, None
