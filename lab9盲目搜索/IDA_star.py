import numpy as np
import time
import sys
sys.setrecursionlimit(1500)

file = open('test.txt', 'w')


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

change = np.array([[0, -1], [1, 0], [-1, 0], [0, 1]])

flag = False
res_before = []


def IDAstar(Maze, end_point, before, MAX_H, now_point, nowcost, estimatecost):
    global flag
    global res_before
    if flag:
        return
    #see_path(Maze, list(before.keys())+list(before.values()))
    #print(nowcost, now_point, end_point)
    x = now_point[0]
    y = now_point[1]
    if estimatecost > MAX_H or x < 0 or x >= Maze.shape[
            0] or y < 0 or y >= Maze.shape[1]:
        return
    if now_point == end_point:
        flag = True
        res_before = before
        return
    estimate = {}
    for i in range(4):
        temp_line = now_point[0] + change[i][0]
        temp_col = now_point[1] + change[i][1]
        if temp_line < 0 or temp_line >= Maze.shape[
                0] or temp_col < 0 or temp_col >= Maze.shape[1] or Maze[
                    temp_line][temp_col] == '1' or (temp_line,
                                                    temp_col) in before.keys():
            continue
        estimate[(temp_line, temp_col)] = nowcost + 1 + abs(
            temp_line - end_point[0]) + abs(temp_col - end_point[1])
    if len(estimate) == 0:
        return
    estimate = dict(sorted(estimate.items(), key=lambda item: item[1]))
    for item in estimate:
        point = item
        before[point] = now_point
        before_copy = before.copy()
        IDAstar(Maze, end_point, before_copy, MAX_H, point, nowcost + 1,
                estimate[item])
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
    before = {}
    print(start_point[0], start_point[1])
    num = abs(start_point[0] - end_point[0]) + abs(start_point[1] -
                                                   end_point[1])

    for i in range(68, 69):
        end = end_point
        start = start_point
        before = {}
        IDAstar(Maze, end_point, before, i, start_point, 0, num)
        if not flag:
            print(i, flag)
        else:
            path = get_path(res_before, end_point, start_point, Maze)
            see_path(Maze, path)
            break

    #while True:
