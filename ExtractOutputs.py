import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from LeNet import Net

def list_outputs(net=None, model_name='mnist_lenet'):
    if net == None:
        net = Net()

    # Create the transformation to prepare the image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),  # mean
                (0.3081,)   # std
            )
        ]
    )

    #save test dataset to ./data, and transform images
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #load test set with 4 samples per minibatch and 2 threads
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # create iterator and get the first image from the dataset
    iterator = iter(testloader)
    inputs, labels= next(iterator)

    # Load the trained model
    net.load_state_dict(torch.load('./models/' + model_name + '.pth'))

    # Propagate the input and return the layer outputs
    layer_outputs = net.forward(inputs, return_layer_out=True)

    # iterate throug the layers
    layer_number = 1
    for layer in layer_outputs:
        # detaches gradient parameter, converts tensor to numpy array, and returns first image in batch
        output = layer.detach().numpy()[0]
        # flatten output
        output = output.flatten()
        
        # Print statistics
        print("\nLayer: {}\nMin: {}\nMax: {}\nMean: {}\nStd: {}\nSize:{}".format(layer_number, np.min(output), np.max(output), np.mean(output), np.std(output), output.size))

        # plot histogram
        plt.hist(output)
        plt.show()

        layer_number += 1


if __name__ == '__main__':
    list_outputs()