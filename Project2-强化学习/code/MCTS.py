from copy import deepcopy
from Othello import *
import numpy as np
import math

B = -1


class MCTS(object):
    """
    MC Tree Search used in AlphaZero Algorithm
    """
    def __init__(self, evalNet, MCTSTimes, Cpuct=1):
        '''
        input:
            Cpuct: constant for the UCT
        '''
        super(MCTS, self).__init__()
        self.evalNet = evalNet
        self.MCTSTimes = MCTSTimes
        self.Cpuct = Cpuct
        # 4 parameters mentioned in MCTS algorithm
        # (s,a) is the key, a is the index
        self.N = {}
        self.W = {}
        self.Q = {}
        # this one is different, s is key, the value is a numpy.ndarray
        self.Psa = {}
        # records the probabilities of moves of the states, and used to compute the self.Q
        # record the encountered States
        self.Vs = {}  # valid moves of a state
        self.Ns = {}
        # the times encountered the states, unnecessary, but convenient

    def virtualLoss(self, game):
        LOSS = 10
        actions = list(map(tuple2index, game.getchoices(B)))
        selectAction = np.random.choice(actions)
        print(selectAction)
        gamestr = game.tostr()
        for action in actions:
            self.N[(gamestr, action)] = 1
            if action == selectAction:
                self.Q[(gamestr, action)] = 0
                self.W[(gamestr, action)] = 0
            else:
                self.Q[(gamestr, action)] = -LOSS
                self.W[(gamestr, action)] = -LOSS

    def search(self, game, Tau):
        '''
        expand the tree to a leaf node for self.MCTSTimes times
        input:
            game: Othello(unified to in Black's view)
            Tau: the temperature using in Annealing
        output:
            Pi: probabilities of the actions in the currenct state of the game
        '''
        for _ in range(self.MCTSTimes):
            newGame = deepcopy(game)
            self.expand(newGame)

        gamestr = game.tostr()
        counts = [
            self.N[(gamestr, a)] if (gamestr, a) in self.N else 0
            for a in range(game.getActionSize())
        ]
        if Tau == 0:
            bestAction = np.argmax(counts)
            probs = np.zeros((len(counts)))
            probs[bestAction] = 1
            return probs

        counts = [x**(1. / Tau) for x in counts]
        if np.sum(counts) > 0:
            probs = [x * 1.0 / (np.sum(counts)) for x in counts]
        else:
            probs = np.zeros((len(counts)))
        return probs

    def expand(self, game):
        '''
        expand the tree from the current state to a leaf node, using UCT(UCB) and updating all parameters
        leaf node: a node that is met firstly

        input:
            game: Othello(unified to in Black's view)
        '''
        gamestr = game.tostr()

        if game.isover():
            return -game.get_winner()

        if gamestr not in self.Psa:  
            self.Psa[gamestr], v = self.evalNet(game.board)
            self.Psa[gamestr] = self.Psa[gamestr].detach().cpu().numpy()[0]
            v = v.detach().cpu().numpy()[0][0]
            validMoves = game.getchoices(B)
            validMask = np.zeros((game.getActionSize()))
            if len(validMoves) > 0:
                for mv in validMoves:
                    validMask[tuple2index(mv)] = 1
                self.Psa[gamestr] = self.Psa[gamestr] * validMask
                sumPs = np.sum(self.Psa[gamestr])

                if sumPs > 0:
                    self.Psa[gamestr] /= sumPs
                else:
                    print(
                        "No place to put!"
                    )
                    self.Psa[gamestr] = self.Psa[gamestr] + validMask
                    self.Psa[gamestr] /= np.sum(self.Psa[gamestr])

            self.Vs[gamestr] = validMask
            self.Ns[gamestr] = 0
            return -v

        validMask = self.Vs[gamestr]
        curBest = -float('inf')
        bestAction = -1

        for action in range(game.getActionSize()):
            if validMask[action]:
                if (gamestr, action) in self.Q:
                    uct = self.Q[
                        (gamestr, action
                         )] + self.Cpuct * self.Psa[gamestr][action] * math.sqrt(
                             self.Ns[gamestr]) / (1 + self.N[(gamestr, action)])
                else:
                    uct = self.Cpuct * self.Psa[gamestr][action] * math.sqrt(
                        self.Ns[gamestr])
                if uct > curBest:
                    curBest = uct
                    bestAction = action

        game.play(*index2tuple(bestAction), B)
        game.board = game.board * -1
        v = self.expand(game)
        if (gamestr, bestAction) in self.Q:
            self.N[(gamestr, bestAction)] += 1
            self.W[(gamestr, bestAction)] -= v
            self.Q[(gamestr, bestAction)] = self.W[
                (gamestr, bestAction)] * 1.0 / self.N[(gamestr, bestAction)]
        else:
            self.N[(gamestr, bestAction)] = 1
            self.W[(gamestr, bestAction)] = -v
            self.Q[(gamestr, bestAction)] = self.W[
                (gamestr, bestAction)] * 1.0 / self.N[(gamestr, bestAction)]
        self.Ns[gamestr] += 1

        return -v