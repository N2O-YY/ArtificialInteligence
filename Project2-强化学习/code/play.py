import time, random
from Othello import *
from MCTS import MCTS
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import argparse
import sys

class AiPlayer:
    def __init__(self, net, side, Tau=0, mcts_times=400):
        self.net = net
        self.side = side
        self.Tau = Tau
        self.mcts_times = mcts_times

    def get_move(self, game):    
        mcts = MCTS(self.net, self.mcts_times)
        game.board *= -self.side
        print("time ", end='')
        st = time.time()
        probs = mcts.search(game, 0)
        ed = time.time()
        print('used %fs' % float(ed-st))
        game.board *= -self.side
        if np.sum(probs) > 0:
            # action = np.sum(np.random.rand()>np.cumsum(probs))
            action = np.argmax(probs)
            print('place :', index2tuple(action))
            game.play(*index2tuple(action), self.side)
        else:
            game.play(-1, -1, -1)




class OthelloGame:
    def __init__(self, net, ai_side, Tau=0, mcts_times=100):
        self.ai_player = AiPlayer(net, ai_side, Tau, mcts_times)
        self.game = Othello()
        self.ai_side = ai_side

    def playgame(self):
        side = -1
        while not self.game.isover():
            self.game.show(side)
            print('score: ',self.game.getScore())
            if len(self.game.getchoices(side))!=0:
                if (side == self.ai_side):
                    self.ai_player.get_move(self.game)
                else:
                    while True:
                        try:
                            x, y = input("please input the place").split()
                            print(x, y)
                            x, y = int(x), int(y)
                            if (x, y) in self.game.getchoices(side):
                                self.game.play(x, y, side)
                                break
                        except Exception as e:
                            print("error!", e)
            else:
                print("No where to put")
            side = -side
        print(self.game.getScore())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default='model780.pkl')
    parser.add_argument("--ai_side",type=int,default=1)
    parser.add_argument("--mcts", type=int, default=600)
    args = parser.parse_args()

    net = torch.load(args.load_dir).cuda()
    OthelloGame(net, args.ai_side, 0, args.mcts).playgame()
