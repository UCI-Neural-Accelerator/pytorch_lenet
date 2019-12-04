from LeNet import Net
from Training import train
from Predict import predict


def train_variations(fc=False, kernel=False):
    if fc == True:
        fc_variation()
    if kernel == True:
        kernel_variation()

def fc_variation():
    print('Fully Connected Layer Size Variation')

    # Combinations of fully connected layer sizes
    fc1_size = [120, 10, 200]
    fc2_size = [84, 10, 120]

    # result dictionary
    accuracy = {}

    # Iterate through all the fully connected layer sizes
    for fc1 in fc1_size:
        for fc2 in fc2_size:
            # create an instance of the neural network
            net = Net(fc1_size=fc1, fc2_size=fc2)
            model = 'fc_variation' + str(fc1) + '_' + str(fc2)
            
            # train the model
            train(net, model_name=(model))

            # measure model accuracy with testing dataset
            accuracy[model] = predict(net, model_name=model)

    print(accuracy)


def kernel_variation():
    accuracy_sum = 0
    model_count = 3

    accuracy = {}

    for trained_model in range(model_count):
        net = Net(kernel_size=7, conv_output_size=2)
        model = 'test' + str(trained_model)
        train(net, model_name=model)
        accuracy[trained_model] = predict(net, model_name=model)
        accuracy_sum += accuracy[trained_model]

    accuracy['average_accuracy'] = accuracy_sum / model_count
    print(accuracy)



if (__name__ == '__main__'):
    train_variations()