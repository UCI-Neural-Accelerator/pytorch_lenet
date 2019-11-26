import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # call the init of the nn.Module
        # Create the structure of the LeNet-5 network
        # 1 input image channel, 6 output channels, 5x5 square convultion
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input, 16 output, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)  # Change to follow LeNet 5 model later
        # fully conneted (linear y = Wx +b)
        # 16 6x6 images input, 120 neuron output
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 120 input, 84 output
        self.fc2 = nn.Linear(120, 84)
        # 84 input, 10 output
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        #get max value from each 2x2 window, zero out all negative values 
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        #forward propagation through fully connected layers
        x = x.view(-1,self.num_flat_features)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

    def num_flat_features(self,x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_featurs *= s
        return num_features

net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())