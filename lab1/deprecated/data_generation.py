import numpy as np
import matplotlib.pyplot as plt


"""
Usage:
data = CostumData(), all parameters are set with defaults
data = CostumData(sd_A = 1, ratio=0.1), however distributions can be specified by setting parameters individually
data_points = data.patterns, access data points
data_labels = data.targets, access data labels

"""


class CostumData:
    n = None  # Number of data points
    mean_A = None  # Mean class A, [x_mean, y_mean]
    sd_A = None  # Sigma class A
    mean_B = None  # Mean class B, [x_mean, y_mean]
    sd_B = None  # Sigma class B
    patterns = None  # Data points, shape=(n,2)
    targets = None  # Data labels, shape=(n,1)
    # delta_predictions = []  # list of np.arrays with shape=(n,1)
    # delta_weights = []
    # slp_predictions = []  # list of np.arrays with shape=(n,1)

    def __init__(self, n=100, mean_A=[2.5, 0.5], sd_A=0.3, mean_B=[-2.5, -0.5], sd_B=0.5, ratio=0.5, splitA=False):
        self.n = n
        self.mean_A = mean_A
        self.sd_A = sd_A
        self.mean_B = mean_B
        self.sd_B = sd_B
        self.patterns, self.targets = self.generate_data(ratio, splitA)

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def generate_data(self, ratio, splitA):
        """Gernerates two data clusters after two different multivariate normal distributions.

        Args:
            ratio (float, optional): Determines the ratio of elements in class A to class B. Defaults to 0.5.

        Returns:
            ([patterns], targets): Returns an array of patterns and an of target labels.
        """
        assert 1 >= ratio >= 0
        size_A = int(self.n * ratio)
        size_B = self.n - size_A
        if splitA:
            size_A1 = size_A // 2
            size_A2 = size_A - size_A1
            class_A_1 = np.random.multivariate_normal(
                self.mean_A, np.diag([self.sd_A, self.sd_A]), size=size_A1)
            negative_mean = np.multiply(self.mean_A, [-1, 1])
            class_A_2 = np.random.multivariate_normal(
                negative_mean, np.diag([self.sd_A, self.sd_A]), size=size_A2)
            class_A = np.concatenate((class_A_1, class_A_2))

        else:
            class_A = np.random.multivariate_normal(
                self.mean_A, np.diag([self.sd_A, self.sd_A]), size=size_A)
        class_B = np.random.multivariate_normal(
            self.mean_B, np.diag([self.sd_B, self.sd_B]), size=size_B)

        patterns = np.concatenate((class_A, class_B))
        targets = np.concatenate(
            ([0 for _ in range(size_A)], [1 for _ in range(size_B)]))

        patterns, targets = self.unison_shuffled_copies(patterns, targets)

        return patterns, targets

    def remove_data(self, a_ratio_to_remove, b_ratio_to_remove):
        assert 0 <= a_ratio_to_remove <= 1
        assert 0 <= b_ratio_to_remove <= 1
        a_indices = np.where(self.targets == 0)[0]
        b_indices = np.where(self.targets == 1)[0]
        a_chosen_indices = np.random.choice(a_indices, int(len(
            a_indices) * a_ratio_to_remove), replace=False)
        b_chosen_indices = np.random.choice(b_indices, int(len(
            b_indices) * b_ratio_to_remove), replace=False)
        chosen_indices = np.concatenate([a_chosen_indices, b_chosen_indices])
        new_targets = np.delete(self.targets, chosen_indices)
        new_patterns = np.delete(self.patterns, chosen_indices, axis=0)
        self.targets = new_targets
        self.patterns = new_patterns
        self.n = self.n - len(chosen_indices)

    def remove_subsets_of_A(self, ratio_from_positive=0.2, ratio_from_negative=0.8):
        assert 0 <= ratio_from_positive <= 1
        assert 0 <= ratio_from_negative <= 1
        a_indices = np.where(self.targets == 0)[0]
        pos_indices = np.where(self.patterns[:, 0] > 0)[0]
        neg_indices = np.where(self.patterns[:, 0] < 0)[0]
        a_pos_indices = a_indices[np.in1d(a_indices, pos_indices)]
        a_neg_indices = a_indices[np.in1d(a_indices, neg_indices)]

        a_chosen_pos = np.random.choice(a_pos_indices, int(len(
            a_pos_indices) * ratio_from_positive), replace=False)
        a_chosen_neg = np.random.choice(a_neg_indices, int(len(
            a_neg_indices) * ratio_from_negative), replace=False)
        chosen_indices = np.concatenate([a_chosen_pos, a_chosen_neg])
        new_targets = np.delete(self.targets, chosen_indices)
        new_patterns = np.delete(self.patterns, chosen_indices, axis=0)
        self.targets = new_targets
        self.patterns = new_patterns
        self.n = self.n - len(chosen_indices)
