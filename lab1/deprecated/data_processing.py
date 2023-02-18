import numpy as np
import matplotlib.pyplot as plt
from delta_rule import execute_delta_rule
from slp import perceptron_learning


class DataEngineer:
    data = None
    number_epochs = None
    learning_rate = None

    delta_errors_1 = None
    delta_errors_2 = None
    delta_weights_1 = None
    delta_weights_2 = None
    delta_predictions_1 = None
    delta_predictions_2 = None
    delta_accuracy_1 = None
    delta_accuracy_2 = None
    delta_mse_1 = None
    delta_mse_2 = None

    slp_errors = None
    slp_weights = None
    slp_predictions = None
    slp_accuracy = None
    slp_mse = None

    def __init__(self, data, valdata, number_epochs=200, learning_rate=0.002):
        self.data = data
        self.valdata = valdata
        self.number_epochs = number_epochs
        self.learning_rate = learning_rate

        # --> Delta
        self.delta_errors_1, delta_weights_1, self.delta_predictions_1 = execute_delta_rule(
            self.data.patterns, self.data.targets, self.valdata.patterns, self.valdata.targets, 
            number_epochs, learning_rate, True)
        self.delta_weights_1 = delta_weights_1[-1]

        self.delta_errors_2, delta_weights_2, self.delta_predictions_2 = execute_delta_rule(
            self.data.patterns, self.data.targets, self.valdata.patterns, self.valdata.targets, 
            number_epochs, learning_rate, False)
        self.delta_weights_2 = delta_weights_2[-1]

        # --> slp
        self.slp_weights, self.slp_errors, self.slp_predictions = perceptron_learning(
            self.data.patterns, self.data.targets, self.valdata.patterns, self.valdata.targets, 
            learning_rate, number_epochs)

        # compute accuracies
        print("Batch Delta")
        self.delta_accuracy_1 = self.compute_accuracies(
            self.delta_predictions_1)
        print("Sequential Delta")
        self.delta_accuracy_2 = self.compute_accuracies(
            self.delta_predictions_2)
        print("Classical")
        self.slp_accuracy = self.compute_accuracies(self.slp_predictions)

        # compute mse
        self.delta_mse_1 = self.compute_mse(self.delta_errors_1, 'delta')
        self.delta_mse_2 = self.compute_mse(self.delta_errors_2, 'delta')
        self.slp_mse = self.compute_mse(self.slp_errors, 'slp')

    def compute_accuracies(self, predictions):
        assert self.valdata.n > 0
        accuracies = []
        for prediction in predictions:
            prediction = np.where(prediction < 1, 0, prediction)
            tp = (self.valdata.targets == prediction).sum()
            accuracies.append(tp/self.valdata.n)

        pred = predictions[-1]
        pred = np.where(pred < 1, 0, pred)
        class_A_indices = np.where(self.valdata.targets == 0)
        class_B_indices = np.where(self.valdata.targets == 1)
        predictions_A = pred[class_A_indices]
        predictions_B = pred[class_B_indices]
        
        sensitivity = np.count_nonzero(predictions_B)/class_B_indices[0].shape[0]
        specificity = (class_A_indices[0].shape[0] - np.count_nonzero(predictions_A))/class_A_indices[0].shape[0]
        print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")
        return accuracies

    def compute_mse(self, errors, algorithm):
        assert self.valdata.n > 0
        mses = []
        for error in errors:
            if(algorithm == 'delta'):
                error = error/2
            mse = (1/self.valdata.n) * (error**2).sum()
            mses.append(mse)
        mses_rounded = np.round(np.array(mses), 2)

        return mses_rounded

    def plot_data(self):
        class_A_indices = np.where(self.valdata.targets == 0)
        class_B_indices = np.where(self.valdata.targets == 1)
        patterns_A = self.valdata.patterns[class_A_indices]
        patterns_B = self.valdata.patterns[class_B_indices]

        plt.scatter(patterns_A[:, 0], patterns_A[:, 1],s=2)
        plt.scatter(patterns_B[:, 0], patterns_B[:, 1],s=2)

        plt.show()

    def plot_overview(self):
        """Plots data distributions, decision boundaries, accuracy and mse
        """
        assert self.delta_accuracy_1 != None
        assert self.delta_accuracy_2 != None
        assert self.slp_accuracy != None
        assert self.delta_mse_1.any() != None
        assert self.delta_mse_2.any() != None
        assert self.slp_mse.any() != None
        print(self.delta_accuracy_1[-1],self.delta_accuracy_2[-1],self.slp_accuracy[-1])
        # plot data again for better overview
        class_A_indices = np.where(self.valdata.targets == 0)
        class_B_indices = np.where(self.valdata.targets == 1)
        patterns_A = self.valdata.patterns[class_A_indices]
        patterns_B = self.valdata.patterns[class_B_indices]

        x_1 = np.linspace(-5., 5., 100)

        figure, axis = plt.subplots(2, 2)
        axis[0, 0].scatter(patterns_A[:, 0], patterns_A[:, 1],s=2)
        axis[0, 0].scatter(patterns_B[:, 0], patterns_B[:, 1],s=2)
        axis[0, 0].plot(x_1, (-self.slp_weights[:, 2] -
                              self.slp_weights[:, 0]*x_1)/self.slp_weights[:, 1], label="Slp", color="red")
        axis[0, 0].plot(x_1, (-self.delta_weights_1[2] -
                              self.delta_weights_1[0]*x_1)/self.delta_weights_1[1], label="Delta, Batch", color="blue")
        axis[0, 0].plot(x_1, (-self.delta_weights_2[2] -
                              self.delta_weights_2[0]*x_1)/self.delta_weights_2[1], label="Delta, seq", color="green")
        axis[0, 0].legend(loc="best")
        axis[0, 0].set_title(("Data Distribution - Learning Rate:",
                             self.learning_rate, " Epochs:", self.number_epochs))
        axis[1, 0].set_title("Comparison Accuracy ")
        axis[1, 0].set_ylim([0, 1.05])
        axis[1, 0].plot(self.slp_accuracy, label='Slp', color='red')
        axis[1, 0].plot(self.delta_accuracy_1,
                        label='Delta, Batch', color='blue')
        axis[1, 0].plot(self.delta_accuracy_2,
                        label='Delta, seq', color='green')
        axis[1, 0].legend(loc="best")
        axis[1, 1].set_title("Comparison MSE")
        axis[1, 1].set_ylim([0, 1.05])
        axis[1, 1].plot(self.slp_mse, label='Slp', color='red')
        axis[1, 1].plot(self.delta_mse_1, label='Delta, Batch', color='blue')
        axis[1, 1].plot(self.delta_mse_2, label='Delta, seq', color='green')
        axis[1, 1].legend(loc="best")

        axis[0, 0].set_xlim([-6, 6])
        axis[0, 0].set_ylim([-6, 6])

        plt.show()

    def plot_accuracies(self, overlapping=False):
        assert self.delta_accuracy_1 != None
        assert self.delta_accuracy_2 != None
        assert self.slp_accuracy != None

        # plot data again for better overview
        class_A_indices = np.where(self.valdata.targets == 0)
        class_B_indices = np.where(self.valdata.targets == 1)
        patterns_A = self.valdata.patterns[class_A_indices]
        patterns_B = self.valdata.patterns[class_B_indices]

        x_1 = np.linspace(-5., 5., 100)

        if(overlapping):
            figure, axis = plt.subplots(1, 2)
            axis[0].scatter(patterns_A[:, 0], patterns_A[:, 1],s=2)
            axis[0].scatter(patterns_B[:, 0], patterns_B[:, 1],s=2)
            axis[0].plot(x_1, (-self.slp_weights[:, 2] -
                               self.slp_weights[:, 0]*x_1)/self.slp_weights[:, 1], label="Slp", color="red")
            axis[0].plot(x_1, (-self.delta_weights_1[2] -
                               self.delta_weights_1[0]*x_1)/self.delta_weights_1[1], label="Delta, Batch", color="blue")
            axis[0].plot(x_1, (-self.delta_weights_2[2] -
                               self.delta_weights_2[0]*x_1)/self.delta_weights_2[1], label="Delta, seq", color="green")
            axis[0].legend(loc="best")
            axis[0].set_title(("Data Distribution - Learning Rate:",
                              self.learning_rate, "- Epochs:", self.number_epochs))
            axis[1].set_title("Comparison Accuracy ")
            axis[1].set_ylim([0, 1.05])
            axis[1].plot(self.slp_accuracy, label='Slp', color='red')
            axis[1].plot(self.delta_accuracy_1,
                         label='Delta, Batch', color='blue')
            axis[1].plot(self.delta_accuracy_2,
                         label='Delta, seq', color='green')
            axis[1].legend(loc="best")

        else:
            # Initialise the subplot function using number of rows and columns
            figure, axis = plt.subplots(2, 2)
            axis[0, 0].scatter(patterns_A[:, 0], patterns_A[:, 1],s=2)
            axis[0, 0].scatter(patterns_B[:, 0], patterns_B[:, 1],s=2)
            axis[0, 0].plot(x_1, (-self.slp_weights[:, 2] -
                                  self.slp_weights[:, 0]*x_1)/self.slp_weights[:, 1], label="Slp", color="red")
            axis[0, 0].plot(x_1, (-self.delta_weights_1[2] -
                                  self.delta_weights_1[0]*x_1)/self.delta_weights_1[1], label="Delta, Batch", color="blue")
            axis[0, 0].plot(x_1, (-self.delta_weights_2[2] -
                                  self.delta_weights_2[0]*x_1)/self.delta_weights_2[1], label="Delta, seq", color="green")
            axis[0, 0].legend(loc="best")
            axis[0, 0].set_title(("Data Distribution - Learning Rate:",
                                 self.learning_rate, "- Epochs:", self.number_epochs))
            axis[0, 1].plot(self.slp_accuracy, label='Slp', color='red')
            axis[0, 1].set_ylim([0, 1.05])
            axis[0, 1].set_title("Accuracy SLP")
            axis[1, 0].plot(self.delta_accuracy_1,
                            label='Delta_1', color='blue')
            axis[1, 0].set_ylim([0, 1.05])
            axis[1, 0].set_title("Accuracy Delta-Rule, Batch")
            axis[1, 1].plot(self.delta_accuracy_2,
                            label='Delta_2', color='blue')
            axis[1, 1].set_ylim([0, 1.05])
            axis[1, 1].set_title("Accuracy Delta-Rule, Sequential")

        plt.show()

    def plot_mse(self, overlapping=False):
        assert self.delta_mse_1.any() != None
        assert self.delta_mse_2.any() != None
        assert self.slp_mse.any() != None

        # plot data again for better overview
        class_A_indices = np.where(self.valdata.targets == 0)
        class_B_indices = np.where(self.valdata.targets == 1)
        patterns_A = self.valdata.patterns[class_A_indices]
        patterns_B = self.valdata.patterns[class_B_indices]
        x_1 = np.linspace(-5., 5., 100)

        if(overlapping):
            figure, axis = plt.subplots(1, 2)
            axis[0].set_title(("Learning Rate:",
                              self.learning_rate, "Epochs:", self.number_epochs))
            axis[0].scatter(patterns_A[:, 0], patterns_A[:, 1],s=2)
            axis[0].scatter(patterns_B[:, 0], patterns_B[:, 1],s=2)
            axis[0].plot(x_1, (-self.slp_weights[:, 2] -
                               self.slp_weights[:, 0]*x_1)/self.slp_weights[:, 1], label="Slp", color="red")
            axis[0].plot(x_1, (-self.delta_weights_1[2] -
                               self.delta_weights_1[0]*x_1)/self.delta_weights_1[1], label="Delta, bash", color="blue")
            axis[0].plot(x_1, (-self.delta_weights_2[2] -
                               self.delta_weights_2[0]*x_1)/self.delta_weights_2[1], label="Delta, seq", color="green")
            axis[0].legend(loc="best")
            axis[1].set_title("Comparison MSE")
            axis[1].set_ylim([0, 1.05])
            axis[1].plot(self.slp_mse, label='Slp', color='red')
            axis[1].plot(self.delta_mse_1, label='Delta, bash', color='blue')
            axis[1].plot(self.delta_mse_2, label='Delta, seq', color='green')
            axis[1].legend(loc="best")
        else:
            figure, axis = plt.subplots(2, 2)
            axis[0, 0].set_title(("Data Distribution - Learning Rate:",
                                 self.learning_rate, "- Epochs:", self.number_epochs))
            axis[0, 0].scatter(patterns_A[:, 0], patterns_A[:, 1],s=2)
            axis[0, 0].scatter(patterns_B[:, 0], patterns_B[:, 1],s=2)
            axis[0, 0].plot(x_1, (-self.slp_weights[:, 2] -
                                  self.slp_weights[:, 0]*x_1)/self.slp_weights[:, 1], label="Slp", color="red")
            axis[0, 0].plot(x_1, (-self.delta_weights_1[2] -
                                  self.delta_weights_1[0]*x_1)/self.delta_weights_1[1], label="Delta, bash", color="blue")
            axis[0, 0].plot(x_1, (-self.delta_weights_2[2] -
                                  self.delta_weights_2[0]*x_1)/self.delta_weights_2[1], label="Delta, seq", color="green")
            axis[0, 0].legend(loc="best")
            axis[0, 1].set_title("MSE SLP")
            axis[0, 1].set_ylim([0, 1.05])
            axis[0, 1].plot(self.slp_mse, label='Slp', color='red')
            axis[1, 0].set_title("MSE Delta-Rule, Bash")
            axis[1, 0].set_ylim([0, 1.05])
            axis[1, 1].set_title("MSE Delta-Rule, Sequential")
            axis[1, 1].set_ylim([0, 1.05])
            axis[1, 0].plot(self.delta_mse_1, label='Delta_1', color='blue')
            axis[1, 1].plot(self.delta_mse_2, label='Delta_2', color='blue')
        plt.show()
