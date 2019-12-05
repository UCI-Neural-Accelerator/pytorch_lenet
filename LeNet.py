import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, kernel_size=5, conv_output_size=4, fc1_size=120, fc2_size=84):
        super(Net, self).__init__() # call the init of the nn.Module
        # Create the structure of the LeNet-5 network
        # 1 input image channel, 6 output channels, 5x5 square convultion
        self.conv1 = nn.Conv2d(1, 6, kernel_size)
        # 6 input, 16 output, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, kernel_size)  # Change to follow LeNet 5 model later
        # fully conneted (linear y = Wx +b)
        # 16 6x6 images input, 120 neuron output
        self.fc1 = nn.Linear(16 * conv_output_size * conv_output_size, fc1_size)
        # 120 input, 84 output
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # 84 input, 10 output
        self.fc3 = nn.Linear(fc2_size, 10)


    def forward(self, x, return_layer_out=False):
        #get max value from each 2x2 window, zero out all negative values
        conv1_out = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        conv2_out = F.avg_pool2d(F.relu(self.conv2(conv1_out)), 2)
        #flatten the matrix to a single array
        flat = conv2_out.view(-1, self.num_flat_features(conv2_out))
        #forward propagation through fully connected layers
        #using relu activation function
        fc1_out = F.relu(self.fc1(flat))
        fc2_out = F.relu(self.fc2(fc1_out))
        output = self.fc3(fc2_out)
        if return_layer_out == True:
            return conv1_out, conv2_out, fc1_out, fc2_out, output
        else:
            return output


    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        

if __name__ == "__main__":
    pass