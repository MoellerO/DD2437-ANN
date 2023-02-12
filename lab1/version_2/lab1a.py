import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def generate_data(n=6):
    '''Returns patterns and targets in shape with a column for each sample. Also adds column for bias term'''
    mA = [2.0, 0.5]
    mB = [-2.0, 0.0]
    sigmaA = 1.0
    sigmaB = 1.0
    ratio = 0.5  # propotion of samples of class A
    size_A = int(n * ratio)
    size_B = n - size_A

    class_A = np.random.multivariate_normal(
        mA, np.diag([sigmaA, sigmaA]), size=size_A)
    class_B = np.random.multivariate_normal(
        mB, np.diag([sigmaB, sigmaB]), size=size_B)

    patterns = np.c_[np.concatenate((class_A, class_B)), np.ones(n)]
    targets = np.concatenate(
        ([-1 for _ in range(size_A)], [1 for _ in range(size_B)]))

    p = np.random.permutation(len(patterns))

    return patterns[p].T, targets[p].T


def create_batches(batch_size, patterns, targets):
    idx = 0
    batches = []
    input_size = patterns.shape[1]
    while idx < input_size:
        if idx + batch_size > input_size - 1:
            end = input_size - 1
            next_idx = input_size
        else:
            end = idx + batch_size
            next_idx = end

        batch = [patterns[:, idx:end],
                 targets[idx: end]]
        batches.append(batch)
        idx = next_idx
    return batches


def perceptron_learning(patterns, targets, weights):
    weighted_sum = np.matmul(weights, patterns)
    prediction = np.where(weighted_sum > 0, 1, -1)
    e = (targets - prediction)[np.newaxis, :]
    return e


def delta_rule(patterns, targets, weights):
    weighted_sum = np.matmul(weights, patterns)
    # prediction = np.where(weighted_sum > 0, 1, 0)[0]
    e = (targets - weighted_sum)[np.newaxis, :]
    return e


def plt_data(patterns, targets):
    class_A_indices = np.where(targets == -1)[0]
    class_B_indices = np.where(targets == 1)[0]

    patterns_A = patterns[:, class_A_indices]
    patterns_B = patterns[:, class_B_indices]
    plt.subplot(1, 2, 1)
    plt.scatter(patterns_A[0, :], patterns_A[1, :], s=2)
    plt.scatter(patterns_B[0, :], patterns_B[1, :], s=2)

    plt.subplot(1, 2, 2)
    plt.scatter(patterns_A[0, :], patterns_A[1, :], s=2)
    plt.scatter(patterns_B[0, :], patterns_B[1, :], s=2)


def plot_decision_boundary(i, patterns, targets, W_perc=None):

    if W_perc is None:
        W_perc = weights_perc[i, :]
        W_delt = weights_delt[i, :]

    x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    z_perc = W_perc[0] * x1 + W_perc[1] * x2 + W_perc[2]
    z_delt = W_delt[0] * x1 + W_delt[1] * x2 + W_delt[2]

    plt.clf()

    plt_data(patterns, targets)

    plt.subplot(1, 2, 1)
    plt.contourf(x1, x2, z_perc, levels=[-100, 0, 100],
                 colors=['blue', 'red'], alpha=0.1)
    plt.contour(x1, x2, z_perc, levels=[0], colors=['black'], linewidths=2)
    plt.title("Perceptron Rule")

    plt.subplot(1, 2, 2)
    plt.contourf(x1, x2, z_delt, levels=[-100, 0, 100],
                 colors=['blue', 'red'], alpha=0.1)
    plt.contour(x1, x2, z_delt, levels=[0], colors=['black'], linewidths=2)
    plt.title("Delta Rule")
    plt.suptitle("Epoch: " + str(i + 1))


def on_close(event):
    event.event_source.stop()
    plt.close()


def main():
    NMB_EPOCHS = 20
    ETA = 0.001
    BATCH_SIZE = 10

    patterns, targets = generate_data(n=100)
    batches = create_batches(BATCH_SIZE, patterns, targets)

    global weights_perc
    global weights_delt
    weights_perc = np.zeros((NMB_EPOCHS, 3))
    weights_perc[0, :] = np.random.normal(size=(3,))
    weights_delt = np.copy(weights_perc)

    for epoch in range(NMB_EPOCHS):
        e_perc_epoch = 0
        e_delt_epoch = 0
        for batch in batches:
            pattern_batch = batch[0]
            target_batch = batch[1]
            e_perc = perceptron_learning(
                pattern_batch, target_batch, weights_perc[epoch, :])
            e_delt = delta_rule(
                pattern_batch, target_batch, weights_delt[epoch, :])
            d_w_perc = ETA * np.dot(e_perc, pattern_batch.T).reshape(3,)
            d_w_delt = ETA * np.dot(e_delt, pattern_batch.T).reshape(3,)
            weights_perc[epoch, :] += d_w_perc
            weights_delt[epoch, :] += d_w_delt
            e_perc_epoch += np.sum(e_perc)
            e_delt_epoch += np.sum(e_delt)

        if e_perc_epoch == 0:
            print("Perceptron Learning converged at", epoch)
        if np.abs(e_delt_epoch) < 0.00001:
            print("Delta Learning converged at", epoch, "with", e_delt_epoch)

        if epoch < NMB_EPOCHS - 1:
            weights_perc[epoch + 1, :] = weights_perc[epoch, :]
            weights_delt[epoch + 1, :] = weights_delt[epoch, :]

    ani = FuncAnimation(plt.gcf(), plot_decision_boundary,
                        frames=NMB_EPOCHS, repeat=False, fargs=(patterns, targets))
    plt.gcf().canvas.mpl_connect(ani, on_close)
    plt.show()


main()
