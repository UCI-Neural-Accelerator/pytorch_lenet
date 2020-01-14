import torch
from LeNet import Net

def adjust_weights(model_name="mnist_lenet", weight_factor=1):
    # load trained model
    net = Net()
    net.load_state_dict(torch.load('./models/' + model_name + '.pth'))

    print(next(param for param in net.fc1.parameters()).data)

    # multiplies old weights by factor
    new_weights = weight_factor * next(param for param in net.fc1.parameters()).data
    next(param for param in net.fc1.parameters()).data = new_weights

    torch.save(net.state_dict(), './models/' + model_name + '_adjusted' + '.pth')

    print(next(param for param in net.fc1.parameters()).data)


if (__name__ == '__main__'):
    adjust_weights(model_name='mnist_lenet_top', weight_factor=4)