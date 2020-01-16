import matplotlib.pyplot as plt
import numpy as np

val_acc = []
val_loss = []
test_acc = []
test_loss = []
file = open('out_CNN.txt', 'r')
lines = file.readlines()
i = 0
index = 1

for line in lines:
    line_div = line.split('/')
    if line_div[0] == '50000':
        line = line.strip('\n')
        line_list = line.split('-')
        val_loss.append(float(line_list[2][-7:]))
        val_acc.append(float(line_list[3][-7:]))
        test_loss.append(float(line_list[4][-7:]))
        test_acc.append(float(line_list[5][-7:]))
x = np.arange(len(val_loss))
fig, ax = plt.subplots()
ax.plot(x, val_loss, color='red')
ax.set(xlabel='episode', ylabel='val_loss', title='val_loss for every episode')
ax.grid()
fig.savefig("val_loss.png")

x = np.arange(len(val_acc))
fig, ax = plt.subplots()
ax.plot(x, val_acc, color='orange')
ax.set(xlabel='episode', ylabel='val_acc', title='val_acc for every episode')
ax.grid()
fig.savefig("val_acc.png")

x = np.arange(len(test_loss))
fig, ax = plt.subplots()
ax.plot(x, test_loss, color='green')
ax.set(xlabel='episode',
       ylabel='test_loss',
       title='test_loss for every episode')
ax.grid()
fig.savefig("test_loss.png")

print(test_acc)
x = np.arange(len(test_acc))
fig, ax = plt.subplots()
ax.plot(x, test_acc, color='purple')
ax.set(xlabel='episode',
       ylabel='test_acc',
       title='test_acc for every episode')
ax.grid()
fig.savefig("test_acc.png")
