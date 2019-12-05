import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from LeNet import Net

def list_weights(net):
    w = torch.flatten(list(net.conv1.parameters())[0]).detach().numpy()
    print(w)
    print("Min: {}\nMax: {}\nStd: {}\nSize:{}".format(np.min(w), np.max(w), np.std(w), w.size))
    plt.hist(w)
    plt.show()
    
    w = torch.flatten(list(net.conv2.parameters())[0]).detach().numpy()
    print(w)
    print("Min: {}\nMax: {}\nStd: {}\nSize:{}".format(np.min(w), np.max(w), np.std(w), w.size))
    plt.hist(w)
    plt.show()


    w = torch.flatten(list(net.fc1.parameters())[0]).detach().numpy()
    print(w)
    print("Min: {}\nMax: {}\nStd: {}\nSize:{}".format(np.min(w), np.max(w), np.std(w), w.size))
    plt.hist(w)
    plt.show()

    w = torch.flatten(list(net.fc2.parameters())[0]).detach().numpy()
    print(w)
    print("Min: {}\nMax: {}\nStd: {}\nSize:{}".format(np.min(w), np.max(w), np.std(w), w.size))
    plt.hist(w)
    plt.show()

    w = torch.flatten(list(net.fc3.parameters())[0]).detach().numpy()
    print(w)
    print("Min: {}\nMax: {}\nStd: {}\nSize:{}".format(np.min(w), np.max(w), np.std(w), w.size))
    plt.hist(w)
    plt.show()


if __name__ is '__main__':
    list_weights()