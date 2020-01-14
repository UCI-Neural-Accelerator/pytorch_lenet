import torch
import torchvision
import torchvision.transforms as transforms

from LeNet import Net
from ExtractWeights import list_weights
from ExtractOutputs import list_outputs

def predict(net=None, model_name='mnist_lenet', get_weights=False, get_outputs=False):
    # check if net was passed in
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

    # Load the trained model
    net.load_state_dict(torch.load('./models/' + model_name + '.pth'))

    correct = 0
    total = 0
    with torch.no_grad():   # disable the gradient calculation for speedup
        for data in testloader: # iterate through the samples
            images, labels = data
            output = net(images)
            # get the indicies of maximum values (same as the number in our case) of the prediction
            _, prediction = torch.max(output.data, 1)
            # increments the total and the correct if the prediction matches the label
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

    print('Accuracy on %d test images: %.2f %%' % (total, 100 * correct / total))

    if get_weights == True:
        print('Print weights\n')
        list_weights(net)

    if get_outputs == True:
        print('Print outputs\n')
        list_outputs(net)

    return (100 * correct / total)


if (__name__ == '__main__'):
    predict(get_outputs=True, get_weights=True, model_name='mnist_lenet_top')