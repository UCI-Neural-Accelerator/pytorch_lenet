import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from LeNet import Net

def train(net: Net, model_name='mnist_lenet'):
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

    #save training dataset to ./data, downloads and trasforms images
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #load train set, with 4 samples per minibatch, randomize images and use 2 threads
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    #print the neural network
    print(net)

    criterion = nn.CrossEntropyLoss() # creating a mean squared error object (try other loss functions)
    # set up optimizer for SGD and pass network parameters and learning rate
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(2):  # iterate over dataset multiple times

        running_loss = 0.0  # running total of the cost function for output
        # iterate through the training set
        for i, data in enumerate(trainloader, 0):
            # get the input and the label
            inputs, labels = data

            optimizer.zero_grad()  # zeros the gradient buffers

            output = net(inputs)   # propogate input through the neural network
            loss = criterion(output, labels) # Puts output and target into criterion object, MSE
            loss.backward()  # calculate the gradient of each parameter based on the MSE loss function
            optimizer.step()  # Does the update
            #print statistics of training
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, (running_loss / 200)))
                running_loss = 0.0

    print('--~~ Finished Training ~~--\n')

    print('Saving Model')
    torch.save(net.state_dict(), './models/' + model_name + '.pth')

if (__name__ == '__main__'):
    net = Net()
    train(net)
