from LeNet import Net
from Training import train
from Predict import predict


def train_variations():
    # Combinations of fully connected layer sizes
    fc1_size = [120, 10, 200]
    fc2_size = [84, 10, 120]

    accuracy = {}

    # Iterate through all the fully connected layer sizes
    for fc1 in fc1_size:
        for fc2 in fc2_size:
            # create an instance of the neural network
            net = Net(fc1, fc2)
            model = 'test_' + str(fc1) + '_' + str(fc2)
            
            # train the model
            train(net, model_name=(model))

            # measure model accuracy with testing dataset
            accuracy[model] = predict(net, model_name=model)

    print(accuracy)


if (__name__ == '__main__'):
    train_variations()