import os, time, random
from Othello import *
from MCTS import MCTS
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

N_HIDDEN = 5


def cal_loss(my_value, labels, my_probs, rollout_prob):
    return torch.mean(
        ((my_value -
          torch.Tensor(labels.astype(float)).reshape(-1, 1))**2) -
        torch.log(my_probs +
                  1e-7).mm(torch.t(torch.Tensor(rollout_prob))).
        gather(1,
               torch.range(0, 127).reshape(-1, 1).long()))


def initialweights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(
            m, nn.BatchNorm2d):
        nn.init.normal(m.weight.data, 0, 2)


class Block1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, in_):
        out_ = torch.nn.functional.relu(self.bn1(self.conv1(in_)))
        out_ = self.bn2(self.conv2(out_))
        out_ += self.shortcut(in_)
        out_ = torch.nn.functional.relu(out_)
        return out_


class Block2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Block2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, in_):
        out_ = torch.nn.functional.relu(self.bn1(self.conv1(in_)))
        out_ = torch.nn.functional.relu(self.bn2(self.conv2(out_)))
        out_ = self.bn3(self.conv3(out_))
        out_ += self.shortcut(in_)
        out_ = torch.nn.functional.relu(out_)
        return out_


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.add_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.add_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self.add_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self.add_layer(block, 512, num_blocks[3], stride=1)
        self.conv_policy = nn.Conv2d(512, 2, 1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, 64)

        self.conv_value = nn.Conv2d(512, 1, 1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value_1 = nn.Linear(1 * 8 * 8, 32)
        self.fc_value_2 = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def add_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, in_):
        if len(in_.shape) == 2:
            in_ = torch.Tensor(in_[np.newaxis, np.newaxis, :, :])
        else:
            in_ = torch.Tensor(in_)
        in_ = in_
        out_ = torch.nn.functional.relu(self.bn1(self.conv1(in_)))
        out_ = self.layer1(out_)
        out_ = self.layer2(out_)
        out_ = self.layer3(out_)
        in_ = self.layer4(out_)

        policy = torch.nn.functional.relu(self.bn_policy(self.conv_policy(in_)))
        policy = policy.view(-1, 8 * 8 * 2)
        policy = torch.nn.functional.softmax(self.fc_policy(policy))
        v = torch.nn.functional.relu(self.bn_value(self.conv_value(in_)))
        v = v.view(-1, 8 * 8 * 1)
        v = torch.nn.functional.relu(self.fc_value_1(v))
        v = torch.nn.functional.tanh(self.fc_value_2(v))
        return policy, v


def OthelloNet():
    return ResNet(Block1, [2, 2, 2, 2])


def get_equal_board(s0, prob, w, m):
    s0_ = np.rot90(s0)
    prob_ = np.rot90(np.array(prob).reshape(8, 8))
    for _ in range(m // 2):
        s0_ = np.rot90(s0_)
        prob_ = np.rot90(prob_)
    if m % 2 == 0:
        s0_ = s0_.T
        prob_ = prob_.T
    prob_ = prob_.reshape((64, ))
    return s0_, prob_, w


def self_play(i, net):
    st = time.time()
    net.optimizer.zero_grad()
    batch_size = 128
    States = []
    game = Othello()
    mcts = MCTS(net, 1000)
    mcts.virtualLoss(game)
    side = -1
    Tau = 1
    while not game.isover():
        game.board *= -side
        probability = mcts.search(game, Tau)
 
        States.append([game.board.copy(), probability, side])

        if np.sum(probability) > 0:
            action = np.sum(np.random.rand() > np.cumsum(probability))
            game.board *= -side
            game.play(*index2tuple(action), side)
        else:
            game.play(-1, -1, -1)

        side = -side

    winner = game.get_winner()

    for state, _ in enumerate(States):
        States[state][2] *= -winner

    expand_data = []
    for s in States:
        for func_index in np.random.permutation(7)[:2]:
            expand_data.append(get_equal_board(s[0], s[1], s[2], func_index))
    np.random.shuffle(expand_data)
    batch_data = np.concatenate(
        [States, expand_data[:batch_size - len(States)]], axis=0)
    inputs = np.concatenate(batch_data[:, 0]).reshape(-1, 8,
                                                      8)[:, np.newaxis, :, :]
    rollout_prob = np.concatenate(batch_data[:, 1]).reshape(-1, 64)
    labels = batch_data[:, 2]

    my_probs, my_value = net(inputs)
    loss = cal_loss(my_value, labels, my_probs, rollout_prob)
    net.optimizer.zero_grad()  # clear gradients for next train
    loss.backward(retain_graph=True)
    net.optimizer.step()

    ed = time.time()
    print("%6d game, time=%4.4fs, loss = %5.5f" % (i, ed - st, float(loss)))
    return inputs, rollout_prob, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", type=int, default=800)
    parser.add_argument("--start_iter", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default="./log.txt")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = OthelloNet()
    if args.start_iter != -1:
        model = torch.load("model" + str(args.start_iter) +
                           ".pkl")
    else:
        model.apply(initialweights)
    print(model)
    for rank in range(args.start_iter + 1, args.start_iter + args.iter_num):
        self_play(rank, model)
        if rank % 20 == 0:
            torch.save(model, 'model' + str(rank) + '.pkl')
    print("Finish!")
