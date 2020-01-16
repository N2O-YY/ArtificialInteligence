import numpy as np
import time
import sys
sys.setrecursionlimit(1500)


def get_maze():
    """
    得到地图
    """
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

flag = False
res_before = []

def IDDFS(Maze, end_point, visited, before, num, now_point):
    ##f.write(str(now_point)+'\n')
    global flag
    global res_before
    if flag:
        return
    x = now_point[0]
    y = now_point[1]
    if x < 0 or x >= Maze.shape[0] or y < 0 or y >= Maze.shape[1] or num < 0:
        #see_path(Maze, visited)
        return
    if now_point == end_point:
        flag = True
        res_before = before
        return 
    for i in range(4):
        temp_line = now_point[0] + change[i][0]
        temp_col = now_point[1] + change[i][1]
        if temp_line < 0 or temp_line >= Maze.shape[
                0] or temp_col < 0 or temp_col >= Maze.shape[1] or Maze[
                    temp_line][temp_col] == '1' or (temp_line,
                                                    temp_col) in visited:
            continue
        point = (temp_line, temp_col)
        before[point] = now_point
        visited.append(point)
        visited_copy = visited.copy()
        before_copy = before.copy()
        IDDFS(Maze, end_point, visited_copy, before_copy, num-1, point)
        visited.remove(point)
        before.pop(point)


def get_path(before, end_point, start_point, Maze):
    res = []
    temp_point = end_point
    while True:
        point = before[temp_point]
        res.append(point)
        temp_point = point
        if point == start_point:
            return res
        Maze[point] = '2'


def see_path(Maze, path):
    """
    路径的可视化
    """
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
    before = {}
    print(start_point[0], start_point[1])
    distance[start_point] = 0
    num = abs(start_point[0] - end_point[0]) + abs(start_point[1] -
                                                   end_point[1])
    f = open('test.txt', 'w')
    for i in range(num, num+30):
        end = end_point
        start = start_point
        before = {}
        visited = []
        IDDFS(Maze, end, visited, before, i, start)
        if not flag:
            print(i, flag)
        else:
            path = get_path(res_before, end_point, start_point, Maze)
            see_path(Maze, path)
            break
     
        

    #while True:
