import sys
import argparse

train_batch_in_each_epoch = 10

filename = 'syn_output.txt'
with open(filename,'r') as fin:
    training_loss_in_batch = []
    for i in range(0,250):
        temp = 0.0
        for j in range(0, train_batch_in_each_epoch):
            line = fin.readline().split() # e.g., ['Epoch:', '[12][138/300]', 'Loss', '0.000740', 'UnNormalizedLoss', '29.6026']
            # print(line)
            temp = temp + float(line[2])
        training_loss_in_batch.append(temp/train_batch_in_each_epoch)

import matplotlib.pyplot as plt
import numpy as np
import os
print(training_loss_in_batch)
training_loss_in_batch = np.array(training_loss_in_batch)

x_test = np.arange(0.0, len(training_loss_in_batch), 1.0)
plt.figure(figsize=(5.5,4))
plt.plot(x_test, training_loss_in_batch,label='Training')

plt.xlim([0.0,250.0])
plt.ylim([0.0,0.15])
plt.legend()
plt.xlabel("Epochs", fontsize=15, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.ylabel("Loss", fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')

plt.savefig("Loss.jpg", bbox_inches='tight')
