import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def generate_data(n=6):
    '''Returns patterns and targets in shape with a column for each sample. Also adds column for bias term'''
    mA = [5.0, 0.5]
    mB = [3.0, 0.0]
    sigmaA = 0.5
    sigmaB = 0.5
    ratio = 0.5  # propotion of samples of class A
    size_A = int(n * ratio)
    size_B = n - size_A

    class_A = np.random.multivariate_normal(
        mA, np.diag([sigmaA, sigmaA]), size=size_A)
    class_B = np.random.multivariate_normal(
        mB, np.diag([sigmaB, sigmaB]), size=size_B)

    patterns = np.c_[np.concatenate((class_A, class_B)), np.ones(n)]
    if REMOVE_BIAS:
        patterns = np.concatenate((class_A, class_B))
    targets = np.concatenate(
        ([-1 for _ in range(size_A)], [1 for _ in range(size_B)]))

    p = np.random.permutation(len(patterns))

    return patterns[p].T, targets[p].T


def generate_splitted_data(n=6):
    '''Returns patterns and targets in shape with a column for each sample. Also adds column for bias term'''
    mA = [5.0, 0.5]
    mB = [3.0, 0.0]
    sigmaA = 0.5
    sigmaB = 0.5
    ratio = 0.5  # propotion of samples of class A
    size_A = int(n * ratio)
    size_A1 = size_A // 2
    size_A2 = size_A - size_A1
    size_B = n - size_A

    class_A1 = np.random.multivariate_normal(
        mA, np.diag([sigmaA, sigmaA]), size=size_A1)
    negative_mA = np.multiply(mA, [-1, 1])

    class_A2 = np.random.multivariate_normal(
        negative_mA, np.diag([sigmaA, sigmaA]), size=size_A2)
    class_A = np.concatenate((class_A1, class_A2))

    class_B = np.random.multivariate_normal(
        mB, np.diag([sigmaB, sigmaB]), size=size_B)

    patterns = np.c_[np.concatenate((class_A, class_B)), np.ones(n)]
    if REMOVE_BIAS:
        patterns = np.concatenate((class_A, class_B))
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
    plt.subplot(2, 2, 1)
    plt.scatter(patterns_A[0, :], patterns_A[1, :], s=2)
    plt.scatter(patterns_B[0, :], patterns_B[1, :], s=2)

    plt.subplot(2, 2, 2)
    plt.scatter(patterns_A[0, :], patterns_A[1, :], s=2)
    plt.scatter(patterns_B[0, :], patterns_B[1, :], s=2)


def plt_accs(perc_accuracy_train, perc_accuracy_val, delt_accuracy_train, delt_accuracy_val):
    plt.subplot(2, 2, 3)
    plt.plot(perc_accuracy_train, label='Test Accuracy')
    plt.plot(perc_accuracy_val, label='Validation Accuracy')
    plt.xlabel("Index")
    plt.ylabel("Accuracy")
    plt.title("Test and Validation Accuracy for Perceptron Rule")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(delt_accuracy_train, label='Test Accuracy')
    plt.plot(delt_accuracy_val, label='Validation Accuracy')
    plt.xlabel("Index")
    plt.ylabel("Accuracy")
    plt.title("Test and Validation Accuracy for Delta Rule")
    plt.legend()


def plot_decision_boundary(i, patterns, targets, weights_perc, weights_delt, perc_accuracy_train, perc_accuracy_val, delt_accuracy_train, delt_accuracy_val, ):

    W_perc = weights_perc[i, :]
    W_delt = weights_delt[i, :]

    x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

    if REMOVE_BIAS:
        z_perc = W_perc[0] * x1 + W_perc[1] * x2
        z_delt = W_delt[0] * x1 + W_delt[1] * x2
    else:
        z_perc = W_perc[0] * x1 + W_perc[1] * x2 + W_perc[2]
        z_delt = W_delt[0] * x1 + W_delt[1] * x2 + W_delt[2]

    plt.clf()

    plt_data(patterns, targets)
    plt_accs(perc_accuracy_train, perc_accuracy_val,
             delt_accuracy_train, delt_accuracy_val)

    plt.subplot(2, 2, 1)
    plt.contourf(x1, x2, z_perc, levels=[-100, 0, 100],
                 colors=['blue', 'red'], alpha=0.1)
    plt.contour(x1, x2, z_perc, levels=[0], colors=['black'], linewidths=2)
    plt.title("Perceptron Rule")

    plt.subplot(2, 2, 2)
    plt.contourf(x1, x2, z_delt, levels=[-100, 0, 100],
                 colors=['blue', 'red'], alpha=0.1)
    plt.contour(x1, x2, z_delt, levels=[0], colors=['black'], linewidths=2)
    plt.title("Delta Rule")
    if REMOVE_BIAS:
        plt.suptitle("Epoch: " + str(i + 1) + ", No Bias")
    else:
        plt.suptitle("Epoch: " + str(i + 1))


def on_close(event):
    event.event_source.stop()
    plt.close()


def comp_acc(patterns, targets, perc_weights, delta_weights):
    if SPLITTED_A:
        val_patterns, val_targets = generate_splitted_data(n=100)
    else:
        val_patterns, val_targets = generate_data(n=100)

    # perc train acc
    perc_weighted_sum = np.matmul(perc_weights, patterns)
    perc_prediction = np.where(perc_weighted_sum > 0, 1, -1)
    perc_equal_entries = np.sum(perc_prediction == targets)
    perc_accuracy_train = perc_equal_entries / len(targets) * 100
    # perc val acc
    val_perc_weighted_sum = np.matmul(perc_weights, val_patterns)
    val_perc_prediction = np.where(val_perc_weighted_sum > 0, 1, -1)
    val_perc_equal_entries = np.sum(val_perc_prediction == val_targets)
    perc_accuracy_val = val_perc_equal_entries / len(val_targets) * 100

    # delta train acc
    delt_weighted_sum = np.matmul(delta_weights, patterns)
    delt_prediction = np.where(delt_weighted_sum > 0, 1, -1)
    delt_equal_entries = np.sum(delt_prediction == targets)
    delt_accuracy_train = delt_equal_entries / len(targets) * 100
    # delta val acc
    val_delt_weighted_sum = np.matmul(delta_weights, val_patterns)
    val_delt_prediction = np.where(val_delt_weighted_sum > 0, 1, -1)
    val_delt_equal_entries = np.sum(val_delt_prediction == val_targets)
    delt_accuracy_val = val_delt_equal_entries / len(val_targets) * 100

    # print("Train Perc Accuracy: {:.2f}%".format(perc_accuracy_train))
    # print("Val Perc Accuracy: {:.2f}%".format(perc_accuracy_val))
    # print("Train Delta Accuracy: {:.2f}%".format(delt_accuracy_train))
    # print("Val Delta Accuracy: {:.2f}%".format(delt_accuracy_val))

    return perc_accuracy_train, perc_accuracy_val, delt_accuracy_train, delt_accuracy_val


def remove_data(patterns, targets, ratio_a, ratio_b):
    a_indices = np.where(targets == -1)[0]
    b_indices = np.where(targets == 1)[0]

    a_num = int(len(a_indices) * ratio_a)
    b_num = int(len(b_indices) * ratio_b)

    a_selected_indices = np.random.choice(
        a_indices, a_num, replace=False)
    b_selected_indices = np.random.choice(
        b_indices, b_num, replace=False)

    remaining_indices = np.array(
        list(set(np.arange(targets.size)) - set(a_selected_indices) - set(b_selected_indices)))
    return patterns[:, remaining_indices], targets[remaining_indices]


def remove_subset_a(patterns, targets, ratio_pos, ratio_neg):
    a_indices = np.where(targets == -1)[0]
    positive_indices = np.where(patterns[0, :] > 0)[0]
    negative_indices = np.where(patterns[0, :] < 0)[0]
    a_pos_indices = a_indices[np.in1d(a_indices, positive_indices)]
    a_neg_indices = a_indices[np.in1d(a_indices, negative_indices)]
    a_chosen_pos = np.random.choice(a_pos_indices, int(len(
        a_pos_indices) * ratio_pos), replace=False)
    a_chosen_neg = np.random.choice(a_neg_indices, int(len(
        a_neg_indices) * ratio_neg), replace=False)
    indices_to_del = np.concatenate([a_chosen_pos, a_chosen_neg])
    new_targets = np.delete(targets, indices_to_del)
    new_patterns = np.delete(patterns, indices_to_del, axis=1)
    return new_patterns, new_targets


def main():
    NMB_EPOCHS = 20
    ETA = 0.001
    BATCH_SIZE = 100
    EPS = 0.001
    REMOVE_A = 0.  # to delte
    REMOVE_B = 0.  # to delte
    REMOVE_POS = 0.8  # to delte
    REMOVE_NEG = 0.2  # to delte

    global REMOVE_BIAS, SPLITTED_A
    REMOVE_BIAS = False
    SPLITTED_A = True

    if SPLITTED_A:
        patterns, targets = generate_splitted_data(n=100)
    else:
        patterns, targets = generate_data(n=100)

    patterns, targets = remove_data(patterns, targets, REMOVE_A, REMOVE_B)
    print(patterns.shape)
    patterns, targets = remove_subset_a(
        patterns, targets, REMOVE_POS, REMOVE_NEG)
    print(patterns.shape)

    batches = create_batches(BATCH_SIZE, patterns, targets)

    # global weights_perc
    # global weights_delt
    weights_perc = np.zeros((NMB_EPOCHS, 3))
    weights_perc[0, :] = np.random.normal(size=(3,))
    if REMOVE_BIAS:
        weights_perc = np.zeros((NMB_EPOCHS, 2))
        weights_perc[0, :] = np.random.normal(size=(2,))
    weights_delt = np.copy(weights_perc)

    perc_accs_train = []
    perc_accs_val = []
    delt_accs_train = []
    delt_accs_val = []
    for epoch in range(NMB_EPOCHS):
        d_perc_epoch = 0
        d_delt_epoch = 0

        for batch in batches:
            pattern_batch = batch[0]
            target_batch = batch[1]
            e_perc = perceptron_learning(
                pattern_batch, target_batch, weights_perc[epoch, :])
            e_delt = delta_rule(
                pattern_batch, target_batch, weights_delt[epoch, :])

            if REMOVE_BIAS:
                d_w_perc = ETA * np.dot(e_perc, pattern_batch.T).reshape(2,)
                d_w_delt = ETA * np.dot(e_delt, pattern_batch.T).reshape(2,)
            else:
                d_w_perc = ETA * np.dot(e_perc, pattern_batch.T).reshape(3,)
                d_w_delt = ETA * np.dot(e_delt, pattern_batch.T).reshape(3,)
            weights_perc[epoch, :] += d_w_perc
            weights_delt[epoch, :] += d_w_delt
            d_perc_epoch += np.sum(d_w_perc)
            d_delt_epoch += np.sum(d_w_delt)

        perc_accuracy_train, perc_accuracy_val, delt_accuracy_train, delt_accuracy_val = comp_acc(patterns, targets,
                                                                                                  weights_perc[epoch, :], weights_delt[epoch, :])
        perc_accs_train.append(perc_accuracy_train)
        perc_accs_val.append(perc_accuracy_val)
        delt_accs_train.append(delt_accuracy_train)
        delt_accs_val.append(delt_accuracy_val)

        if np.abs(d_perc_epoch) <= EPS:
            print("Perceptron Learning converged at", epoch)
        if np.abs(d_delt_epoch) < EPS:
            print("Delta Learning converged at", epoch, "with", d_delt_epoch)

        if epoch < NMB_EPOCHS - 1:
            weights_perc[epoch + 1, :] = weights_perc[epoch, :]
            weights_delt[epoch + 1, :] = weights_delt[epoch, :]

    ani = FuncAnimation(plt.gcf(), plot_decision_boundary,
                        frames=NMB_EPOCHS, repeat=False, fargs=(patterns, targets, weights_perc, weights_delt, perc_accs_train, perc_accs_val, delt_accs_train, delt_accs_val))
    plt.gcf().canvas.mpl_connect(ani, on_close)
    plt.show()


main()
