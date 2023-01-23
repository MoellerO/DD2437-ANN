from data_generation import CostumData
from data_processing import DataEngineer
from delta_rule import execute_delta_rule
from slp import perceptron_learning
import matplotlib.pyplot as plt

import numpy as np


np.random.seed(0)
# General: For comparison use: number or ratio of misclassified examples/mean squared error at each epoch

# 3.1.1:
# - Generate two classes (100dp) x
# - shuffle data x
# - plot patterns with different colors per class x
# data = CostumData()
data_1 = CostumData(mean_A=[4.5, 0.5], mean_B=[-4.5, -0.5],
                    sd_A=1, sd_B=1)  # Very easy linear separable
# data_2 = CostumData(mean_A=[1.5, 0.5], mean_B=[-1.5, -0.5],
#                     sd_A=0.3, sd_B=0.5)  # Linear separable
# data_3 = CostumData(mean_A=[1.5, 0.5], mean_B=[-1.5, -0.5],
#                     sd_A=2, sd_B=2)  # Not linear separable
# engineer_1 = DataEngineer(data_1, number_epochs=100, learning_rate=0.0004)
# engineer_2 = DataEngineer(data_2, 100, 0.0004)
# engineer_3 = DataEngineer(data_3, 100, 0.0004)

# 3.1.2: Compare percepton learning with delta in batch
# Task: Ratio correctly classified & mse per epoch

# plot overview: MSE, accuracies and decision boundaries
# engineer_1.plot_overview()
# engineer_2.plot_overview()
# engineer_3.plot_overview()

# plot accuracies - you dont really need those if you use plot_overview
# engineer_1.plot_accuracies(overlapping=True)
# engineer_2.plot_accuracies(overlapping=True)
# engineer_3.plot_accuracies(overlapping=True)

# # plot MSE - you dont really need those if you use plot_overview
# engineer_1.plot_mse(overlapping=True)
# engineer_2.plot_mse(overlapping=True)
# engineer_3.plot_mse(overlapping=True)


# Task: Ajdust learning rate and study convergence of the two algorithms
# Answer: - Choose data_2, linear separable but with closer class means
#         - Choose learning rates: 0.5, 0.1, 0.01, 0.001
# learning rate to small, algorithms converge to fast
#lr_engineer_1 = DataEngineer(data_2, learning_rate=0.01)
#lr_engineer_2 = DataEngineer(data_2, learning_rate=0.001)
#lr_engineer_3 = DataEngineer(data_2, learning_rate=0.0001)
# learning rate to high, algorithms dont converge
#lr_engineer_4 = DataEngineer(data_2, learning_rate=0.00001)
# lr_engineer_1.plot_overview()
# lr_engineer_2.plot_overview()
# lr_engineer_3.plot_overview()
# lr_engineer_4.plot_overview()
# engineer_1.plot_overview()

# TODO:
#     + How sensitive is learning to random initialization
#     + Remove Bias, train network with delta in batch mode
#     + What cases would the perceptron without bias converge and classify correctly all samples?


# TODO: 3.1.3
# remove 25% each class, 50% A, 50% B
# Comment in relevant codelines and repeat program execution. Random seed guarentees comparable results
# Parameter splitA forms data according to the description in the assignment 3.1.3
# Other tasks defined in 3.1.3


# np.random.seed(42)
# new_data = CostumData(splitA=True, mean_A=[0.0, 0.0], mean_B=[-1.0, -1.0],
#                       sd_A=1.0, sd_B=1.0)
# engineer_before_removing = DataEngineer(
#     new_data, new_data, number_epochs=1000, learning_rate=0.0001)
# # engineer_before_removing.plot_overview()

# # --> Comment in line to remove
# # new_data.remove_data(a_ratio_to_remove=0.25, b_ratio_to_remove=0.25)
# #new_data.remove_data(a_ratio_to_remove=0.5, b_ratio_to_remove=0.0)
# #new_data.remove_data(a_ratio_to_remove=0.0, b_ratio_to_remove=0.5)
# new_data.remove_subsets_of_A(ratio_from_positive=0.2, ratio_from_negative=0.8)

# np.random.seed(42)
# all_data = CostumData(splitA=True, mean_A=[0.0, 0.0], mean_B=[-1.0, -1.0],
#                       sd_A=1.0, sd_B=1.0)

# engineer_after_removing = DataEngineer(
#     new_data, all_data, number_epochs=1000, learning_rate=0.0001)
# engineer_after_removing.plot_overview()
