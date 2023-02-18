import numpy as np
from data_generation import CostumData
import matplotlib.pyplot as plt


def seq_train(patterns, targets, valx, valy, weights, lr):
    num_samples = patterns.shape[1]
    # tot_error = 0.0
    errors = []
    for n in range(num_samples):
        xn = patterns[:, n]
        tn = targets[n]
        error = np.dot(weights, xn) - tn
        dw = -lr * error * xn.T
        weights = weights + dw

    for n in range(valy.shape[0]):
        xn = valx[:, n]
        tn = valy[n]
        errors.append(np.dot(weights, xn) - tn)

    return np.array(errors), weights


def batch_train(patterns, targets, valx, valy, weights, lr):
    error = np.dot(weights, patterns) - targets
    dw = - lr * np.dot(error, patterns.T)
    print(dw)
    weights = weights + dw

    error = np.dot(weights, valx) - valy
    return error, weights


def predict(patterns, weights):
    pred = np.dot(weights, patterns)
    pred = np.where(pred < 0, -1.0, 1.0)
    return pred


def train_loop(epochs, patterns, targets, valx, valy,
               weights, lr, batch=True):
    errors = []
    weights_ = []
    predictions = []  # only used to compute acc and mse later.
    for i in range(epochs):
        if batch:
            error, weights = batch_train(patterns,
                                         targets, valx, valy, weights, lr)
        else:
            error, weights = seq_train(
                patterns, targets, valx, valy, weights, lr)
        prediction = predict(valx, weights)
        predictions.append(prediction)
        errors.append(error)
        weights_.append(weights)
    return errors, weights_, predictions


def evaluate(true, pred):
    return (true == pred).sum()


def plot_decision_boundary(weights, patterns, targets):
    x_1 = np.linspace(-5., 5., 100)

    fig, ax = plt.subplots()
    ax.plot(x_1, (-weights[2]-weights[0]*x_1)/weights[1])

    ax.set_xlim((-5., 5.))
    ax.set_ylim((-5., 5.))

    class_A_indices = np.where(targets == -1)
    class_B_indices = np.where(targets == 1)
    patterns_A = patterns[class_A_indices]
    patterns_B = patterns[class_B_indices]

    plt.scatter(patterns_A[:, 0], patterns_A[:, 1])
    plt.scatter(patterns_B[:, 0], patterns_B[:, 1])

    plt.show()

############################################


def execute_delta_rule(patterns, targets, valx, valy, number_epochs, learning_rate, batch=True):
    X = patterns
    T = targets
    T = np.where(T < 1, -1.0, T)
    X = X.T
    ones = np.ones([1, X.shape[1]])
    X = np.vstack([X, ones])
    w = np.random.normal(size=3)

    valx = valx.T
    valx = np.concatenate((valx, np.ones((1, valy.shape[0]))),
                          axis=0)  # add a fixed row of 1's

    # print(evaluate(T, predict(X, w)))

    # Batch training's lr should be 100 times smaller
    # than the equivalent sequential!!!
    errors, weights, predictions = train_loop(
        number_epochs, X, T, valx, valy, w, learning_rate, batch=batch)
    # print(errors[-1])
    w = weights[-1]

    # print(errors)
    # print(predict(X, w))
    # print(evaluate(T, predict(X, w)))

    # plot_decision_boundary(w, x_orig, T)
    return errors, weights, predictions
