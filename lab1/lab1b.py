import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def generate_data(n=6):
    '''Returns patterns and targets in shape with a column for each sample. Also adds column for bias term'''
    # mA = [3.0, 0.5]
    # mB = [-3.0, 0.5]
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
    # mA = [4.0, 0.5]
    # mB = [-4.0, 0.5]
    # sigmaA = 0.5
    # sigmaB = 0.5
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
    weighted_sum = np.dot(weights, patterns)
    prediction = np.where(weighted_sum > 0, 1.0, -1.0)
    e = (targets - prediction)[np.newaxis, :]
    return e


def delta_rule(patterns, targets, weights):
    weighted_sum = np.dot(weights, patterns)
    # prediction = np.where(weighted_sum > 0, 1, 0)[0]
    e = (targets - weighted_sum)[np.newaxis, :]
    return e


def plt_data(patterns, targets):
    class_A_indices = np.where(targets == -1)[0]
    class_B_indices = np.where(targets == 1)[0]

    patterns_A = patterns[:, class_A_indices]
    patterns_B = patterns[:, class_B_indices]
    plt.subplot(1, 2, 1)
    plt.scatter(patterns_A[0, :], patterns_A[1, :], s=2, label='Class A')
    plt.scatter(patterns_B[0, :], patterns_B[1, :], s=2, label='Class B')
    # plt.legend()


def plt_accs(delt_accuracy_train, delt_accuracy_val):
    plt.subplot(1, 2, 2)
    plt.plot(delt_accuracy_train, label='Seq Train Accuracy')
    if PLOT_VAL:
        plt.plot(delt_accuracy_val, label='Seq Validation Accuracy')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.title("Test and Validation Accuracy for Delta Rule")
    plt.legend()


def plt_mses(p_mses, d_mses, fd_mses):
    plt.plot(p_mses, label='Perc')
    plt.plot(d_mses, label='Delta-Seq')
    plt.plot(fd_mses, label='Delta-Batch')
    plt.title("MSE for delta and perceptron rule")

    plt.legend()


def plot_decision_boundary(i, patterns, targets, weights_delt, delt_accuracy_train, delt_accuracy_val):

    W_delt = weights_delt[i, :]

    x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

    if REMOVE_BIAS:
        z_delt = W_delt[0] * x1 + W_delt[1] * x2
    else:
        z_delt = W_delt[0] * x1 + W_delt[1] * x2 + W_delt[2]

    plt.clf()

    plt_data(patterns, targets)
    plt_accs(
        delt_accuracy_train, delt_accuracy_val)

    plt.subplot(1, 2, 1)
    plt.scatter(0, 0, marker='x', s=100, color='red')
    plt.contourf(x1, x2, z_delt, levels=[-100, 0, 100],
                 colors=['blue', 'red'], alpha=0.1)
    plt.contour(x1, x2, z_delt, levels=[0], colors=[
                'black'], linewidths=2, label='Seq')
    plt.legend()
    plt.title("Delta Rule")
    if REMOVE_BIAS:
        plt.suptitle("Epoch: " + str(i + 1) + ", No Bias")
    else:
        plt.suptitle("Epoch: " + str(i + 1))


def on_close(event):
    event.event_source.stop()
    plt.close()


def comp_mses(patterns, targets, wp, wd, fwd):
    perc_weighted_sum = np.dot(wp, patterns)
    mse_p = np.mean((targets - perc_weighted_sum) ** 2)

    delt_weighted_sum = np.dot(wd, patterns)
    mse_d = np.mean((targets - delt_weighted_sum) ** 2)

    full_delt_weighted_sum = np.dot(fwd, patterns)
    mse_fd = np.mean((targets - full_delt_weighted_sum) ** 2)

    return mse_p, mse_d, mse_fd


def comp_acc(patterns, targets, delta_weights):
    if SPLITTED_A:
        val_patterns, val_targets = generate_splitted_data(n=100)
    else:
        val_patterns, val_targets = generate_data(n=100)

    # delta train acc
    delt_weighted_sum = np.dot(delta_weights, patterns)
    delt_prediction = np.where(delt_weighted_sum > 0, 1.0, -1.0)
    delt_equal_entries = np.sum(delt_prediction == targets)
    delt_accuracy_train = delt_equal_entries / len(targets) * 100
    # delta val acc
    val_delt_weighted_sum = np.dot(delta_weights, val_patterns)
    val_delt_prediction = np.where(val_delt_weighted_sum > 0, 1.0, -1.0)
    val_delt_equal_entries = np.sum(val_delt_prediction == val_targets)
    delt_accuracy_val = val_delt_equal_entries / len(val_targets) * 100

    # print("Train Delta Accuracy: {:.2f}%".format(delt_accuracy_train))
    # print("Val Delta Accuracy: {:.2f}%".format(delt_accuracy_val))

    return delt_accuracy_train, delt_accuracy_val


def comp_class_acc(patterns, targets, perc_weights, delta_weights, full_weights_delt):
    if SPLITTED_A:
        val_patterns, val_targets = generate_splitted_data(n=100)
    else:
        val_patterns, val_targets = generate_data(n=100)

    # perc train acc
    perc_weighted_sum = np.dot(perc_weights, patterns)
    perc_prediction = np.where(perc_weighted_sum > 0, 1.0, -1.0)
    class1_acc_train = np.sum(np.logical_and(
        perc_prediction == 1, targets == 1)) / np.sum(targets == 1) * 100
    class_neg1_acc_train = np.sum(np.logical_and(
        perc_prediction == -1, targets == -1)) / np.sum(targets == -1) * 100

    # perc val acc
    val_perc_weighted_sum = np.dot(perc_weights, val_patterns)
    val_perc_prediction = np.where(val_perc_weighted_sum > 0, 1.0, -1.0)
    class1_acc_val = np.sum(np.logical_and(
        val_perc_prediction == 1, val_targets == 1)) / np.sum(val_targets == 1) * 100
    class_neg1_acc_val = np.sum(np.logical_and(
        val_perc_prediction == -1, val_targets == -1)) / np.sum(val_targets == -1) * 100

    # delta train acc
    delt_weighted_sum = np.dot(delta_weights, patterns)
    delt_prediction = np.where(delt_weighted_sum > 0, 1.0, -1.0)
    class1_delt_acc_train = np.sum(np.logical_and(
        delt_prediction == 1, targets == 1)) / np.sum(targets == 1) * 100
    class_neg1_delt_acc_train = np.sum(np.logical_and(
        delt_prediction == -1, targets == -1)) / np.sum(targets == -1) * 100

    # delta val acc
    val_delt_weighted_sum = np.dot(delta_weights, val_patterns)
    val_delt_prediction = np.where(val_delt_weighted_sum > 0, 1.0, -1.0)
    class1_delt_acc_val = np.sum(np.logical_and(
        val_delt_prediction == 1, val_targets == 1)) / np.sum(val_targets == 1) * 100
    class_neg1_delt_acc_val = np.sum(np.logical_and(
        val_delt_prediction == -1, val_targets == -1)) / np.sum(val_targets == -1) * 100

    # full batch delta train acc
    full_delt_weighted_sum = np.dot(full_weights_delt, patterns)
    full_delt_prediction = np.where(full_delt_weighted_sum > 0, 1.0, -1.0)
    class1_full_delt_acc_train = np.sum(np.logical_and(
        full_delt_prediction == 1, targets == 1)) / np.sum(targets == 1) * 100
    class_neg1_full_delt_acc_train = np.sum(np.logical_and(
        full_delt_prediction == -1, targets == -1)) / np.sum(targets == -1) * 100

    # full batch delta vall acc
    full_delt_weighted_sum = np.dot(full_weights_delt, val_patterns)
    full_delt_prediction = np.where(full_delt_weighted_sum > 0, 1.0, -1.0)
    class1_full_delt_acc_val = np.sum(np.logical_and(
        full_delt_prediction == 1, val_targets == 1)) / np.sum(val_targets == 1) * 100
    class_neg1_full_delt_acc_val = np.sum(np.logical_and(
        full_delt_prediction == -1, val_targets == -1)) / np.sum(val_targets == -1) * 100

    return class1_acc_train, class_neg1_acc_train, class1_acc_val, class_neg1_acc_val, class1_delt_acc_train, class_neg1_delt_acc_train, class1_delt_acc_val, class_neg1_delt_acc_val, class1_full_delt_acc_train, class_neg1_full_delt_acc_train, class1_full_delt_acc_val, class_neg1_full_delt_acc_val


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


def plot_class_accuracies(class_accs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(class_accs[0], label=f"B-train")
    axes[0].plot(class_accs[1], label=f"A-train")

    axes[0].set_title("Perceptron")

    axes[1].plot(class_accs[4], label=f"B-train")
    axes[1].plot(class_accs[5], label=f"A-train")

    axes[1].set_title("Delta Seq")

    axes[2].plot(class_accs[8], label=f"B-train")
    axes[2].plot(class_accs[9], label=f"A-train")

    axes[2].set_title("Delta Batch")

    if PLOT_VAL:
        axes[0].plot(class_accs[2], label=f"B-val")
        axes[0].plot(class_accs[3], label=f"A-val")
        axes[1].plot(class_accs[6], label=f"B-val")
        axes[1].plot(class_accs[7], label=f"A-val")
        axes[2].plot(class_accs[10], label=f"B-val")
        axes[2].plot(class_accs[11], label=f"A-val")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
    plt.tight_layout()


# Constants
NMB_EPOCHS = 100
ETA = 0.0005
N = 100
BATCH_SIZE = 1
REMOVE_A = 0.0
REMOVE_B = 0.0
REMOVE_POS = 0.2
REMOVE_NEG = 0.8
mA = [1.0, 0.3]  # mean class A
mB = [0.0, -0.1]  # mean class B
sigmaA = 0.2
sigmaB = 0.3
REMOVE_BIAS = False
SPLITTED_A = False
PLOT_VAL = False
REMOVE_DATA = False


def main():
    # Generate data
    if SPLITTED_A:
        patterns, targets = generate_splitted_data(n=N)
    else:
        patterns, targets = generate_data(n=N)

    # Preprocessing
    if REMOVE_DATA:
        patterns, targets = remove_data(patterns, targets, REMOVE_A, REMOVE_B)
        patterns, targets = remove_subset_a(
            patterns, targets, REMOVE_POS, REMOVE_NEG)

    # Create batches
    batches = create_batches(BATCH_SIZE, patterns, targets)

    # Initialize weights
    weights_delt = np.zeros((NMB_EPOCHS, 3))
    weights_delt[0, :] = np.random.normal(size=(3,))
    if REMOVE_BIAS:
        weights_delt = np.zeros((NMB_EPOCHS, 2))
        weights_delt[0, :] = np.random.normal(size=(2,))

    # Train and validate the model
    delt_accs_train, delt_accs_val = train_model(
        patterns, targets, batches, weights_delt)

    # Plot the decision boundary
    ani = FuncAnimation(plt.gcf(), plot_decision_boundary,
                        frames=NMB_EPOCHS, repeat=False, fargs=(patterns, targets, weights_delt, delt_accs_train, delt_accs_val))
    plt.gcf().canvas.mpl_connect(ani, on_close)
    plt.show()


def train_model(patterns, targets, batches, weights_delt):
    delt_accs_train = []
    delt_accs_val = []

    for epoch in range(NMB_EPOCHS):
        d_delt_epoch = 0

        for batch in batches:
            pattern_batch = batch[0]
            target_batch = batch[1]

            # Delta rule
            e_delt = delta_rule(
                pattern_batch, target_batch, weights_delt[epoch, :])

            # Calculate weight updates
            if REMOVE_BIAS:
                d_w_delt = ETA * np.dot(e_delt, pattern_batch.T).reshape(2,)
            else:
                d_w_delt = ETA * np.dot(e_delt, pattern_batch.T).reshape(3,)

            # Update weights
            weights_delt[epoch, :] += d_w_delt
            d_delt_epoch += np.sum(d_w_delt)

        # Compute accuracy on training and validation sets
        delt_accuracy_train, delt_accuracy_val = comp_acc(
            patterns, targets, weights_delt[epoch, :])

        delt_accs_train.append(delt_accuracy_train)
        delt_accs_val.append(delt_accuracy_val)

        if epoch < NMB_EPOCHS - 1:
            weights_delt[epoch + 1, :] = weights_delt[epoch, :]

    return delt_accs_train, delt_accs_val


main()
