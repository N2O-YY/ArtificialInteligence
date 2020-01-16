import numpy as np
import time
import sys
#coding=utf-8
sys.setrecursionlimit(15000)
#f = open('test.txt', 'w')


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


change = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])



def bi_search(Maze, now_start_point, now_end_point, start_visited, end_visited,
              start_before, end_before, turn, start_distance, end_distance):
    """
    参数的含义：
    Maze：list，地图
    now_start_point：tuple，从起点出发的当前的节点
    now_end_point：tuple，从终点出发的当前的节点
    start_visited：list，从起点出发所经过的节点
    end_visited：list，从终点出发所经过的节点
    start_before：dic，从起点出发的每个点的前继
    end_before：dic，从终点出发的每个点的前继
    turn：bool，表示该从起点那端出发还是终点那端出发
    start_distance：dic，代价
    end_distance：dic，代价
    """
    if now_start_point == now_end_point or now_start_point in end_visited or now_end_point in start_visited:
        if now_start_point == now_end_point:
            return now_start_point
        elif now_start_point in end_visited:
            return now_start_point
        elif now_end_point in start_visited:
            return now_end_point
    if turn is True:
        if len(start_distance) > 0:
            now_point = list(start_distance.keys())[0]
            dis = list(start_distance.values())[0]
            #f.write(str(now_point) + '\n')
            start_distance.pop(now_point)
            start_visited.append(now_point)
            for i in range(4):
                temp_line = now_point[0] + change[i][0]
                temp_col = now_point[1] + change[i][1]
                if temp_line < 0 or temp_line >= Maze.shape[
                        0] or temp_col < 0 or temp_col >= Maze.shape[
                            1] or Maze[temp_line][temp_col] == '1' or (
                                temp_line, temp_col) in start_visited:
                    continue
                start_distance[(temp_line, temp_col)] = dis + 1
                start_before[(temp_line, temp_col)] = now_point
            start_distance = dict(
                sorted(start_distance.items(),
                       key=lambda x: x[1],
                       reverse=False))
            return bi_search(Maze, now_point, now_end_point, start_visited,
                             end_visited, start_before, end_before, not turn,
                             start_distance, end_distance)
        else:
            return bi_search(Maze, now_start_point, now_end_point,
                             start_visited, end_visited, start_before,
                             end_before, not turn, start_distance,
                             end_distance)
    else:
        if len(end_distance) > 0:
            now_point = list(end_distance.keys())[0]
            dis = list(end_distance.values())[0]
            #f.write(str(now_point) + '\n')
            end_distance.pop(now_point)
            end_visited.append(now_point)
            for i in range(4):
                temp_line = now_point[0] + change[i][0]
                temp_col = now_point[1] + change[i][1]
                if temp_line < 0 or temp_line >= Maze.shape[
                        0] or temp_col < 0 or temp_col >= Maze.shape[
                            1] or Maze[temp_line][temp_col] == '1' or (
                                temp_line, temp_col) in end_visited:
                    continue
                end_distance[(temp_line, temp_col)] = dis + 1
                end_before[(temp_line, temp_col)] = now_point
            end_distance = dict(
                sorted(end_distance.items(), key=lambda x: x[1],
                       reverse=False))
            return bi_search(Maze, now_start_point, now_point, start_visited,
                             end_visited, start_before, end_before, not turn,
                             start_distance, end_distance)
        else:
            return bi_search(Maze, now_start_point, now_end_point,
                             start_visited, end_visited, start_before,
                             end_before, not turn, start_distance,
                             end_distance)


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


def see_path(Maze):
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
    start_visited = []
    end_visited = []
    start_before = {}
    end_before = {}
    turn = True
    start_distance = {}
    end_distance = {}
    start_distance[start_point] = 0
    end_distance[end_point] = 0

    print(start_point[0], start_point[1])

    time_start = time.time()

    cross_point = bi_search(Maze, start_point, end_point, start_visited,
                            end_visited, start_before, end_before, turn,
                            start_distance, end_distance)
    #print(Uniform_cost(Maze, end_point, distance, visited, before))
    time_end = time.time()
    Maze[cross_point] = '2'
    print(time_end - time_start)
    res = get_path(start_before, cross_point, start_point, Maze)
    print(res)

    res2 = get_path(end_before, cross_point, end_point, Maze)

    print(res2)

    see_path(Maze)
    #print(before)
    #path = get_path(before, end_point, start_point, Maze)
    #see_path(Maze, path)