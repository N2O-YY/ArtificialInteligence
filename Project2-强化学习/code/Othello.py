#####################################################
##NOTE:这里主要是实现黑白棋的逻辑，以及一些训练时候的特征转换
##-1表示黑棋，1表示白棋
######################################################

import numpy as np
from copy import deepcopy


class Othello(object):
    def __init__(self):
        self.initial()

    def initial(self):
        self.board = np.zeros((8, 8), dtype=np.int)
        self.board[3, 3] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1
        self.board[4, 4] = 1

    def play(self, x, y, side):
        if x == -1 and y == -1:
            return
        self.board[x, y] = side
        self.converse(x, y, side)

    def canput(self, x, y, side):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if (self.validdirection(x, y, side, dx, dy)):
                    return True
        return False

    def isover(self):
        for i in range(8):
            for j in range(8):
                if self.board[i, j] == 0 and (self.canput(i, j, -1)
                                              or self.canput(i, j, 1)):
                    return False
        return True

    def winner(self):
        t = np.sum(self.board)
        if t > 0:
            return 1
        if t < 0:
            return -1
        return 0

    def getchoices(self, side):
        choices = []
        for i in range(8):
            for j in range(8):
                if self.board[i, j] == 0 and self.canput(i, j, side):
                    choices.append((i, j))
        return choices

    def getScore(self):
        black = np.sum(self.board == -1)
        white = np.sum(self.board == 1)
        return black, white

    def validdirection(self, x, y, side, dx, dy):
        tx = x + 2 * dx
        if tx < 0 or tx > 7:
            return False
        ty = y + 2 * dy
        if ty < 0 or ty > 7:
            return False
        if self.board[x + dx, y + dy] != -1 * side:
            return False
        while self.board[tx, ty] != side:
            if self.board[tx, ty] == 0:
                return False
            tx += dx
            ty += dy
            if tx < 0 or tx > 7:
                return False
            if ty < 0 or ty > 7:
                return False
        return True

    def converse(self, x, y, side):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if (self.validdirection(x, y, side, dx, dy)):
                    self.replace(x, y, side, dx, dy)

    def replace(self, x, y, side, dx, dy):
        tx = x + dx
        ty = y + dy
        while self.board[tx, ty] != side:
            self.board[tx, ty] = side
            tx += dx
            ty += dy

    def show(self, side):
        print("   ", end="")
        for i in range(8):
            print("%2d" % (i), end="")
        print("\n   ", end="")
        for _ in range(16):
            print("-", end="")
        print("-")
        for i in range(8):
            print("%2d " % (i), end="")
            for j in range(8):
                print("|" +
                      ("+" if self.canput(i, j, side) and self.board[i, j] == 0
                       else Othello.num2color(self.board[i, j])),
                      end="")
            print("|")
            print("   ", end="")
            for _ in range(16):
                print("-", end="")
            print("-")

    def copy(self):
        return deepcopy(self)

    def tostr(self):
        res = ''
        for v in self.board.flat:
            res += str(v + 1)
        return res

    @staticmethod
    def getActionSize():
        return 64

    def num2color(x):
        if x == 1:
            return 'W'
        elif x == -1:
            return 'B'
        return ' '


def feature2board(fea, side):
    my = fea[0, :, :]
    opp = fea[1, :, :]
    return my * side + opp * (-side)


def board2feature(board, side):
    fea = np.zeros((2, 8, 8))
    fea[0, :, :] = (board == side)
    fea[1, :, :] = (board == -side)
    fea = np.array(fea, dtype=int)
    return fea


def tuple2index(mv):
    return mv[0] * 8 + mv[1]


def index2tuple(mv):
    if mv == -1:
        return (-1, -1)
    return (mv // 8, mv % 8)
