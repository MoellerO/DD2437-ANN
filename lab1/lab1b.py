import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter


def generate_data(n=6):
    '''Returns patterns and targets in shape with a column for each sample. Also adds column for bias term'''
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


def generate_splitted_data(n=6):
    '''Returns patterns and targets in shape with a column for each sample. Also adds column for bias term'''
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
    targets = np.concatenate(
        ([-1 for _ in range(size_A)], [1 for _ in range(size_B)]))

    p = np.random.permutation(len(patterns))

    return patterns[p].T, targets[p].T


def create_batches(batch_size, patterns, targets):
    idx = 0
    batches = []
    input_size = len(targets)
    while idx < input_size:
        if idx + batch_size > input_size - 1:
            end = input_size  # * was input_size -1 before
            next_idx = input_size
        else:
            end = idx + batch_size
            next_idx = end
        batch = [patterns[:, idx:end],
                 targets[idx: end]]
        batches.append(batch)
        idx = next_idx
    return batches


def delta_rule(patterns, targets, weights):
    weighted_sum = np.dot(weights, patterns)
    # prediction = np.where(weighted_sum > 0, 1, 0)[0]
    e = (targets - weighted_sum)[np.newaxis, :]
    return e


def plt_data(patterns, targets, val_data):
    class_A_indices = np.where(targets == -1)[0]
    class_B_indices = np.where(targets == 1)[0]
    patterns_A = patterns[:, class_A_indices]
    patterns_B = patterns[:, class_B_indices]

    if val_data != None:
        val_patterns, val_targets = val_data
        val_class_A_indices = np.where(val_targets == -1)[0]
        val_class_B_indices = np.where(val_targets == 1)[0]
        val_patterns_A = val_patterns[:, val_class_A_indices]
        val_patterns_B = val_patterns[:, val_class_B_indices]

    plt.subplot(1, 3, 1)
    plt.scatter(patterns_A[0, :], patterns_A[1, :],
                s=2, label='Class A', color="blue")
    plt.scatter(patterns_B[0, :], patterns_B[1, :],
                s=2, label='Class B', color="red")
    if val_data != None:
        plt.scatter(val_patterns_A[0, :], val_patterns_A[1,
                    :], s=12, label='Val Class A', marker='x', color="blue")
        plt.scatter(val_patterns_B[0, :], val_patterns_B[1,
                    :], s=12, label='Val Class B', marker='x', color="red")
        # plt.legend()


def plt_accs(accs, batch_accs):
    seq_acc_train, seq_acc_val = accs

    sav_window_length = 7
    sav_polyorder = 2

    # Smooth the lines using the Savitzky-Golay filter
    seq_smooth = savgol_filter(
        seq_acc_train, sav_window_length, sav_polyorder)
    seq_val_smooth = savgol_filter(
        seq_acc_val, sav_window_length, sav_polyorder)

    if batch_accs != None:
        batch_acc_train, batch_acc_val = batch_accs
        batch_smooth = savgol_filter(
            batch_acc_train, sav_window_length, sav_polyorder)
        batch_val_smooth = savgol_filter(
            batch_acc_val, sav_window_length, sav_polyorder)

    plt.subplot(1, 3, 2)
    plt.plot(seq_smooth, label='seq-train', color='blue')
    if batch_accs != None:
        plt.plot(batch_smooth, label='batch-train', color='red')
    if PLOT_VAL:
        plt.plot(seq_val_smooth, label='seq-val',
                 color='blue', linestyle='--', alpha=0.25)
        plt.plot(batch_val_smooth, label='batch-val',
                 color='red', linestyle='--', alpha=0.25)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()


def plt_mses(mse, mse_batch):
    sav_window_length = 7
    sav_polyorder = 2

    seq, seq_val = mse
    if mse_batch != None:
        batch, batch_val = mse_batch
        batch_smooth = savgol_filter(
            batch, sav_window_length, sav_polyorder)
        batch_val_smooth = savgol_filter(
            batch_val, sav_window_length, sav_polyorder)

    # Smooth the lines using the Savitzky-Golay filter
    seq_smooth = savgol_filter(
        seq, sav_window_length, sav_polyorder)
    seq_val_smooth = savgol_filter(
        seq_val, sav_window_length, sav_polyorder)

    plt.subplot(1, 3, 3)
    plt.plot(seq_smooth, label='seq (smoothed)', color='blue')
    if mse_batch != None:
        plt.plot(batch_smooth, label='batch (smoothed)', color='red')
    if PLOT_VAL:
        plt.plot(seq_val_smooth, label='seq-val (smoothed)',
                 color='blue', linestyle='--', alpha=0.25)
        plt.plot(batch_val_smooth, label='batch-val (smoothed)',
                 color='red', linestyle='--', alpha=0.25)

    plt.title("MSE")

    plt.legend()


def plot_gaussian(i, patterns, targets, seq_model):
    W, V, accs, mses = seq_model

    x_test = np.linspace(-5, 5, GRID_SIZE)
    y_test = np.linspace(-5, 5, GRID_SIZE)

    xy = np.array(np.meshgrid(x_test, y_test))
    x_test = xy[0].reshape((GRID_SIZE * GRID_SIZE, 1))
    y_test = xy[1].reshape((GRID_SIZE * GRID_SIZE, 1))

    patterns_test = np.stack((x_test, y_test)).T[0]  # (N,2)
    patterns_test = np.c_[patterns_test, np.ones(len(x_test))].T

    W = W[i, :]
    V = V[i, :]

    _, _, _, z_delt = forward(patterns_test, W, V)

    x = x_test.reshape((GRID_SIZE, GRID_SIZE))
    y = y_test.reshape((GRID_SIZE, GRID_SIZE))
    z = z_delt.reshape((GRID_SIZE, GRID_SIZE))

    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', linewidth=0)
    plt.title("Estimated guassian for epoch: " + str(i))


def plot_decision_boundary(i, patterns, targets, val_data, seq_model, batch_model):

    x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

    X = np.vstack((x1.ravel(), x2.ravel(), np.ones_like(x1.ravel())))

    W, V, accs, mses = seq_model
    W = W[i, :]
    V = V[i, :]
    _, _, _, z_delt = forward(X, W, V)
    # Reshape the output to match the shape of the meshgrid
    z_delt = z_delt.reshape(x1.shape)

    accs_batch = None
    mses_batch = None
    if batch_model != None:
        W_batch, V_batch, accs_batch, mses_batch = batch_model
        W_batch = W_batch[i, :]
        V_batch = V_batch[i, :]
        _, _, _, z_delt_batch = forward(X, W_batch, V_batch)
        z_delt_batch = z_delt_batch.reshape(x1.shape)

    plt.clf()

    plt_data(patterns, targets, val_data)
    plt_accs(
        accs, accs_batch)
    plt_mses(mses, mses_batch)

    plt.subplot(1, 3, 1)
    plt.scatter(0, 0, marker='x', s=100, color='red')
    plt.contourf(x1, x2, z_delt, levels=[-100, 0, 100],
                 colors=['blue', 'red'], alpha=0.1)
    plt.contour(x1, x2, z_delt, levels=[0], colors=[
                'blue'], linewidths=2, label='Seq')
    if batch_model != None:
        plt.contour(x1, x2, z_delt_batch, levels=[0], colors=[
                    'red'], linewidths=2, label='Batch')
    plt.legend()
    plt.title("Two Layer Perceptron")
    plt.suptitle("Epoch: " + str(i + 1))


def on_close(event):
    event.event_source.stop()
    plt.close()


def comp_mses(patterns, targets, val_data, W, V):
    if SPLITTED_A:
        val_patterns, val_targets = generate_splitted_data(n=100)
    else:
        val_patterns, val_targets = generate_data(n=100)

    if val_data != None:
        val_patterns, val_targets = val_data

    # train
    _, _, _, train_weighted_sum = forward(patterns, W, V)
    train_mse = np.mean((targets - train_weighted_sum) ** 2)

    # val
    _, _, _, val_weighted_sum = forward(val_patterns, W, V)
    val_mse = np.mean((val_targets - val_weighted_sum) ** 2)

    return train_mse, val_mse


def comp_acc(patterns, targets, val_data, W, V):
    if SPLITTED_A:
        val_patterns, val_targets = generate_splitted_data(n=100)
    else:
        val_patterns, val_targets = generate_data(n=100)

    if val_data != None:
        val_patterns, val_targets = val_data

    # delta train acc
    _, _, _, delt_weighted_sum = forward(patterns, W, V)
    delt_prediction = np.where(delt_weighted_sum > 0, 1.0, -1.0)
    delt_equal_entries = np.sum(delt_prediction == targets)
    delt_accuracy_train = delt_equal_entries / len(targets) * 100
    # delta val acc
    _, _, _, val_delt_weighted_sum = forward(val_patterns, W, V)
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
    to_delete = np.array(
        list(set(np.arange(targets.size)) - set(remaining_indices)))
    return patterns[:, remaining_indices], targets[remaining_indices], to_delete


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
    return new_patterns, new_targets, indices_to_del


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


def activation(inp):
    return 2. / (1 + np.exp(-inp)) - 1


def deriv_act(inp):
    return ((1 + inp) * (1 - inp)) * 0.5


def forward(patterns, W, V):
    H_in = np.dot(W, patterns)
    H_out = activation(H_in)

    # Add bias as new row
    new_row = np.ones((1, patterns.shape[1]))
    H_out = np.vstack((H_out, new_row))
    O_in = np.dot(V, H_out)
    O_out = activation(O_in)

    return H_in, H_out, O_in, O_out


def backward(H_in, H_out, O_in, O_out, V, targets):
    delta_o = (O_out - targets) * deriv_act(O_out)
    delta_h = (np.dot(V.T, delta_o)) * deriv_act(H_out)
    delta_h = delta_h[:-1]

    return delta_h, delta_o


# Constants - Data Gen
N = 100
mA = [1.0, 0.3]  # mean class A
mB = [0.0, -0.1]  # mean class B
sigmaA = 0.2
sigmaB = 0.3
REMOVE_A = 0.
REMOVE_B = 0.
REMOVE_POS = 0.2
REMOVE_NEG = 0.8
SPLITTED_A = True
REMOVE_DATA = True
PLOT_VAL = False


# Constants - Training
NMB_NODES = 16
NMB_EPOCHS = 100
ETA = 0.001
BATCH_SIZE = 900
ALPHA = 0.9

# Constants - other
ANIMATE = True
GRID_SIZE = 30


def gaussian(xy):
    return np.exp(-(xy[0]**2 + xy[1]**2) / 10) - 0.5


def generate_gauss_data(min=-5, max=5):
    grid = np.linspace(min, max, GRID_SIZE)
    xy = np.meshgrid(grid, grid)
    z = gaussian(xy)
    x = xy[0]
    y = xy[1]

    # plot_gaussian(x, y, z)

    x = x.reshape((GRID_SIZE * GRID_SIZE, 1))
    y = y.reshape((GRID_SIZE * GRID_SIZE, 1))
    z = z.reshape((GRID_SIZE * GRID_SIZE))

    patterns = np.stack((x, y)).T[0]  # (N,2)
    patterns = np.c_[patterns, np.ones(len(x))]
    p = np.random.permutation(len(patterns))

    targets = z

    return patterns[p].T, targets[p].T


def function_approx():
    patterns, targets = generate_gauss_data()
    # Initialize weights
    W = np.zeros((NMB_EPOCHS, NMB_NODES, 3))
    W[0, :] = np.random.normal(size=(NMB_NODES, 3))
    V = np.zeros((NMB_EPOCHS, 1, NMB_NODES + 1))
    V[0, :] = np.random.normal(size=(1, NMB_NODES + 1))

    # Train and validate the model using sequential mode (if batchsize == 1)
    seq_batches = create_batches(BATCH_SIZE, patterns, targets)
    seq_accs, seq_mses = train_model(
        patterns, targets, None, seq_batches, W, V)
    seq_model = (W, V, seq_accs, seq_mses)

    # Plot the decision boundary
    if ANIMATE:
        ani = FuncAnimation(plt.gcf(), plot_gaussian,
                            frames=NMB_EPOCHS, repeat=False, fargs=(patterns, targets, seq_model))
        plt.gcf().canvas.mpl_connect(ani, on_close)
    else:
        plot_gaussian(NMB_EPOCHS - 1, patterns,
                      targets, seq_model)

    plt.show()


def two_layer_perceptron():
    # Generate data
    if SPLITTED_A:
        patterns, targets = generate_splitted_data(n=N)
    else:
        patterns, targets = generate_data(n=N)

    # Preprocessing
    if REMOVE_DATA:
        rem_patterns, rem_targets, remaining_indices1 = remove_data(
            patterns, targets, REMOVE_A, REMOVE_B)
        rem_patterns, rem_targets, remaining_indices2 = remove_subset_a(
            rem_patterns, rem_targets, REMOVE_POS, REMOVE_NEG)

        if remaining_indices1.size == 0:
            remaining_indices = remaining_indices2
        else:
            remaining_indices = np.concatenate(
                (remaining_indices1, remaining_indices2))
        hold_patterns = patterns[:, remaining_indices]
        hold_targets = targets[remaining_indices]
        val_data = [hold_patterns, hold_targets]

        patterns = rem_patterns
        targets = rem_targets
        print(patterns.shape)
        print(targets.shape)
    # Initialize weights
    W = np.zeros((NMB_EPOCHS, NMB_NODES, 3))
    W[0, :] = np.random.normal(size=(NMB_NODES, 3))
    V = np.zeros((NMB_EPOCHS, 1, NMB_NODES + 1))
    V[0, :] = np.random.normal(size=(1, NMB_NODES + 1))
    W_batch = np.copy(W)
    V_batch = np.copy(V)

    # Train and validate the model using sequential mode (if batchsize == 1)
    seq_batches = create_batches(BATCH_SIZE, patterns, targets)
    seq_accs, seq_mses = train_model(
        patterns, targets, val_data, seq_batches, W, V)

    # Train and validate the model using batchmode
    full_batch = create_batches(N, patterns, targets)
    batch_accs, batch_mses = train_model(
        patterns, targets, val_data, full_batch, W_batch, V_batch)

    seq_model = (W, V, seq_accs, seq_mses)
    batch_model = (W_batch, V_batch, batch_accs, batch_mses)

    # Plot the decision boundary
    if ANIMATE:
        ani = FuncAnimation(plt.gcf(), plot_decision_boundary,
                            frames=NMB_EPOCHS, repeat=False, fargs=(patterns, targets, val_data, seq_model, batch_model))
        plt.gcf().canvas.mpl_connect(ani, on_close)
    else:
        plot_decision_boundary(NMB_EPOCHS - 1, patterns,
                               targets, val_data, seq_model, batch_model)

    plt.show()


def train_model(patterns, targets, val_data, batches, W, V):
    seq_accs_train = []
    seq_accs_val = []
    seq_mses_train = []
    seq_mses_val = []

    for epoch in range(NMB_EPOCHS):
        for batch in batches:
            pattern_batch = batch[0]
            target_batch = batch[1]

            # Forward Pass
            H_in, H_out, O_in, O_out = forward(
                pattern_batch, W[epoch, :], V[epoch, :])

            # Backward Pass
            delta_h, delta_o = backward(
                H_in, H_out, O_in, O_out, V[epoch, :], target_batch)

            if epoch == 0:
                dw = np.dot(delta_h, pattern_batch.T)
                dv = np.dot(delta_o, H_out.T)
            else:
                dw = (ALPHA * dw) - ((1 - ALPHA) *
                                     np.dot(delta_h, pattern_batch.T))
                dv = (ALPHA * dv) - ((1 - ALPHA) * np.dot(delta_o, H_out.T))

            W[epoch, :] += ETA * dw
            V[epoch, :] += ETA * dv

        # Compute accuracy on training and validation sets
        seq_acc_train, seq_acc_val = comp_acc(
            patterns, targets, val_data, W[epoch, :], V[epoch, :])
        seq_accs_train.append(seq_acc_train)
        seq_accs_val.append(seq_acc_val)

        # compute MSEs
        seq_train_mse, seq_val_mse = comp_mses(
            patterns, targets, val_data, W[epoch, :], V[epoch, :])
        seq_mses_train.append(seq_train_mse)
        seq_mses_val.append(seq_val_mse)

        if epoch < NMB_EPOCHS - 1:
            W[epoch + 1, :] = W[epoch, :]
            V[epoch + 1, :] = V[epoch, :]

    return [seq_accs_train, seq_accs_val], [seq_mses_train, seq_mses_val]


# two_layer_perceptron()
function_approx()
