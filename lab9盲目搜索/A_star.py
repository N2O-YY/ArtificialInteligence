import numpy as np
import time
import sys
sys.setrecursionlimit(15000)


def get_maze():
    Maze = []
    f = open('MazeData.txt', 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        index = len(Maze)
        Maze.append([])
        for a in line:
            Maze[index].append(a)
    return np.array(Maze)


def get_point(Maze, key):
    (line, col) = Maze.shape
    for i in range(line):
        for j in range(col):
            if Maze[i][j] == key:
                return i, j


#def Uniform_cost(Maze):

change = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def Astar(Maze, end_point, distance, visited, real_distance, before):
    now_point = list(distance.keys())[0]
    dis = real_distance[now_point]
    if now_point == end_point:
        return real_distance[now_point]
    distance.pop(now_point)
    if now_point in visited:
        return Astar(Maze, end_point, distance, visited, real_distance, before)
    visited.append(now_point)
    for i in range(4):
        temp_line = now_point[0] + change[i][0]
        temp_col = now_point[1] + change[i][1]
        if temp_line < 0 or temp_line >= Maze.shape[
                0] or temp_col < 0 or temp_col >= Maze.shape[1] or Maze[
                    temp_line][temp_col] == '1' or (temp_line,
                                                    temp_col) in visited:
            continue
        before[(temp_line, temp_col)] = now_point
        distance[(temp_line,
                  temp_col)] = dis + 1 + abs(temp_line -
                                             end_point[0]) + abs(temp_col -
                                                                 end_point[1])
        real_distance[(temp_line, temp_col)] = dis + 1
    distance = dict(sorted(distance.items(), key=lambda x: x[1],
                           reverse=False))
    return Astar(Maze, end_point, distance, visited, real_distance, before)


def get_path(before, end_point, start_point, Maze):
    res = []
    temp_point = end_point
    while True:
        point = before[temp_point]
        print(point)
        res.append(point)
        temp_point = point
        if point == start_point:
            return res
        Maze[point] = '2'


def see_path(Maze, path):
    for maze in Maze:
        for c in maze:
            if c == 'S' or c == 'E':
                print('\033[0;34m' + c + " " + '\033[0m', end="")
            elif c == '1':
                print('\033[0;;40m' + " " * 2 + '\033[0m', end="")
            elif c == '0':
                print(" " * 2, end="")
            elif c == '2':
                print('\033[0;31m' + "*" + " " + '\033[0m', end="")
        print()


if __name__ == "__main__":
    Maze = get_maze()
    start_point = get_point(Maze, 'S')
    end_point = get_point(Maze, 'E')
    visited = []
    distance = {}
    real_distance = {}
    real_distance[start_point] = 0
    before = {}
    distance[start_point] = 0
    time_start = time.time()
    print(Astar(Maze, end_point, distance, visited, real_distance, before))
    time_end = time.time()
    print((time_end - time_start))
    path = get_path(before, end_point, start_point, Maze)
    see_path(Maze, path)
    