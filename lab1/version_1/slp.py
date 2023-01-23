import numpy as np
import matplotlib.pyplot as plt
from data_generation import CostumData


def perceptron_learning(patterns, targets, valx, valy, learning_rate, epoch_count=1):
    num_of_datapoints = patterns.shape[0]
    weights = np.random.normal(
        loc=0.0, scale=1.0, size=(1, 3))  # random weights
    # X = patterns.reshape(2,num_of_datapoints)
    X = patterns.T
    X = np.concatenate((X, np.ones((1, num_of_datapoints))),
                       axis=0)  # add a fixed row of 1's
    valx = valx.T
    valx = np.concatenate((valx, np.ones((1, valy.shape[0]))),
                          axis=0)  # add a fixed row of 1's
    # X has space (3,num_of_datapoints)
    error_list = []
    prediction_list = []
    for e in range(epoch_count):
        # sequantial mode
        weighted_sums = np.dot(weights, X)
        activations = np.where(weighted_sums > 0, 1, 0)[0]
        # print("Epoch", e)
        # plot_decision_boundary(weights, patterns, targets)

        if(np.array_equal(activations, targets)):
            print("[SLP] All correctly classified")
            error_list.append(np.array([0 for i in range(num_of_datapoints)]))
            prediction_list.append(targets)
            break
        errors = []
        predictions = []
        for i in range(num_of_datapoints):
            sum = np.dot(weights, X[:, i])
            activation = 0
            if(sum > 0):
                activation = 1
            e = targets[i] - activation
            delta_w = learning_rate * e * X[:, i]
            weights += delta_w

        for i in range(valy.shape[0]):
            sum = np.dot(weights, valx[:, i])
            activation = 0
            if(sum > 0):
                activation = 1
            e = valy[i] - activation
            errors.append(e)
            predictions.append(activation)
        prediction_list.append(np.array(predictions))
        error_list.append(np.array(errors))

    # plot_decision_boundary(weights, patterns, targets) TODO
    return weights, error_list, prediction_list


def plot_decision_boundary(weights, patterns, targets):
    x_1 = np.linspace(-5., 5., 100)

    fig, ax = plt.subplots()
    ax.plot(x_1, (-weights[:, 2] - weights[:, 0] * x_1) / weights[:, 1])

    ax.set_xlim((-5., 5.))
    ax.set_ylim((-5., 5.))

    class_A_indices = np.where(targets == 0)
    class_B_indices = np.where(targets == 1)
    patterns_A = patterns[class_A_indices]
    patterns_B = patterns[class_B_indices]

    plt.scatter(patterns_A[:, 0], patterns_A[:, 1])
    plt.scatter(patterns_B[:, 0], patterns_B[:, 1])

    plt.show()


# data = CostumData(sd_A=0.1, sd_B=0.1)
# patterns = data.patterns
# targets = data.targets


# perceptron_learning(patterns, targets, 0.001, 10000)
