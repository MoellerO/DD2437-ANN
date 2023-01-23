import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    n = 100
    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
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
        ([0 for _ in range(size_A)], [1 for _ in range(size_B)]))

    p = np.random.permutation(len(patterns))

    return patterns[p], targets[p]


def plot_data(patterns, targets):
    class_A_indices = np.where(targets == 0)
    class_B_indices = np.where(targets == 1)
    patterns_A = patterns[class_A_indices]
    patterns_B = patterns[class_B_indices]

    plt.scatter(patterns_A[:, 0], patterns_A[:, 1], s=2)
    plt.scatter(patterns_B[:, 0], patterns_B[:, 1], s=2)

    plt.show()


def perceptron_learning():
  


def main():
    patterns, targets = generate_data()
    weights = np.random.normal(size=3)
    plot_data(patterns, targets)


main()
