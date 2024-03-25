from util import *


class RestrictedBoltzmannMachine:
    """
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    def __init__(
        self,
        ndim_visible,
        ndim_hidden,
        is_bottom=False,
        image_size=[28, 28],
        is_top=False,
        n_labels=10,
        batch_size=10,
        learning_rate=0.001,
        momentum=0.5,
        regularization=0.0,
    ):
        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden
        self.is_bottom = is_bottom
        if is_bottom:
            self.image_size = image_size
        self.is_top = is_top
        if is_top:
            self.n_labels = 10
        self.batch_size = batch_size
        self.delta_bias_v = None
        self.delta_weight_vh = None
        self.delta_bias_h = None
        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden)
        )
        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        self.delta_weight_v_to_h = 0
        self.delta_weight_h_to_v = 0
        self.weight_v_to_h = None
        self.weight_h_to_v = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.print_period = 5000
        self.rf = (
            {  # receptive-fields. Only applicable when visible layer is input data
                "period": 5000,  # iteration period to visualize
                "grid": [5, 5],  # size of the grid
                "ids": np.random.randint(
                    0, self.ndim_hidden, 25
                ),  # pick some random hidden units
            }
        )

        return

    def cd1(self, visible_trainset, n_iterations=10000, print_nums=20):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling
        Args:
          visible_trainset: training data for this rbm, shape is (size of training
                            set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns
                        a mini-batch)
        """
        self.rf["period"] = n_iterations // 10
        self.print_period = n_iterations // print_nums
        print("learning CD1")
        n_samples = visible_trainset.shape[0]
        losses = []
        for it in range(n_iterations):
            start_ind = (self.batch_size * it) % n_samples
            end_ind = (self.batch_size * (it + 1)) % n_samples
            v_0 = visible_trainset[start_ind:end_ind, :]
            _, h_0 = self.get_h_given_v(v_0)
            _, v_1 = self.get_v_given_h(h_0)
            _, h_1 = self.get_h_given_v(v_1)
            self.update_params(v_0, h_0, v_1, h_1)

            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(
                    weights=self.weight_vh[:, self.rf["ids"]].reshape(
                        (self.image_size[0], self.image_size[1], -1)
                    ),
                    it=it,
                    grid=self.rf["grid"],
                )

            # print progress
            if it % self.print_period == 0:
                reconstruction = self.get_v_given_h(
                    self.get_h_given_v(visible_trainset)[1]
                )[1]
                recon_loss = (
                    np.sum(np.linalg.norm(reconstruction - visible_trainset, axis=1))
                    / visible_trainset.shape[0]
                )
                print("iteration=%7d recon_loss=%4.4f" % (it, recon_loss))
                losses.append(recon_loss)
        return losses

    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters.
        You could also add weight decay and momentum for weight updates.
        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """
        dot0 = np.dot(v_0.T, h_0)
        dot1 = np.dot(v_k.T, h_k)
        change = self.learning_rate * (dot0 - dot1)
        change_v = self.learning_rate * (v_0 - v_k)
        change_h = self.learning_rate * (h_0 - h_k)
        if self.delta_weight_vh is None:
            self.delta_weight_vh = change
            self.delta_bias_v = change_v
            self.delta_bias_h = change_h
        else:
            self.delta_weight_vh = (
                self.momentum * self.delta_weight_vh + (1 - self.momentum) * change
            )
            self.delta_bias_v = (
                self.momentum * self.delta_bias_v
                + (1 - self.momentum) * self.delta_bias_v
            )
            self.delta_bias_h = (
                self.momentum * self.delta_bias_h
                + (1 - self.momentum) * self.delta_bias_h
            )
        self.bias_v += (
            np.sum(self.delta_bias_v, axis=0) - self.regularization * self.bias_v
        )
        self.weight_vh += self.delta_weight_vh - self.regularization * self.weight_vh
        self.bias_h += (
            np.sum(self.delta_bias_h, axis=0) - self.regularization * self.bias_h
        )
        return

    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)
        Uses undirected weight "weight_vh" and bias "bias_h"
        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None
        n_samples = visible_minibatch.shape[0]
        # probabilities = np.zeros((n_samples, self.ndim_hidden))
        # activations = np.zeros((n_samples, self.ndim_hidden))
        # # print(self.bias_h.shape)
        # # print(visible_minibatch.shape)
        # # print(self.weight_vh.shape)
        # probabilities = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        # sample = np.random.random((n_samples, self.ndim_hidden))
        # activations = np.where(sample < probabilities, 1.0, 0.0)

        p_h_given_v = sigmoid(np.dot(visible_minibatch, self.weight_vh) + self.bias_h)
        h = sample_binary(p_h_given_v)
        return p_h_given_v, h

    def depr_get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)
        Uses undirected weight "weight_vh" and bias "bias_v"
        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        # def softmax(x):
        #     e_x = np.exp(
        #         x - np.max(x, axis=1, keepdims=True)
        #     )  # subtract max(x) for numerical stability
        #     return e_x / e_x.sum(axis=1, keepdims=True)

        assert self.weight_vh is not None
        n_samples = hidden_minibatch.shape[0]
        if self.is_top:
            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \

            total_input = self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T)
            probabilities_data = sigmoid(total_input[:, : -self.n_labels])
            probabilities_labels = softmax(total_input[:, -self.n_labels :])
            probabilities = np.concatenate(
                [probabilities_data, probabilities_labels], axis=1
            )
            activations_data = np.random.binomial(1, probabilities_data)
            activations_labels = np.array(
                [np.random.multinomial(1, p) for p in probabilities_labels]
            )
            activations_labels = activations_labels.reshape(-1, self.n_labels)
            activations = np.concatenate([activations_data, activations_labels], axis=1)

        else:
            probabilities = np.zeros((n_samples, self.ndim_visible))
            activations = np.zeros((n_samples, self.ndim_visible))
            probabilities = sigmoid(
                self.bias_v + np.dot(self.weight_vh, hidden_minibatch.T).T
            )
            sample = np.random.random((n_samples, self.ndim_visible))
            activations = np.where(sample < probabilities, 1.0, 0.0)
        return probabilities, activations

    def get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:
            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # finished
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (
            #  replace the pass below). \ Note that this section can also be postponed until TASK 4.2, since in this
            #  task, stand-alone RBMs do not contain labels in visible layer.

            support = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            support[support < -75] = -75
            p_v_given_h, v = np.zeros(support.shape), np.zeros(support.shape)

            # split into two part and apply different activation functions
            p_v_given_h[:, : -self.n_labels] = sigmoid(support[:, : -self.n_labels])
            p_v_given_h[:, -self.n_labels :] = softmax(support[:, -self.n_labels :])

            v[:, : -self.n_labels] = sample_binary(p_v_given_h[:, : -self.n_labels])
            v[:, -self.n_labels :] = sample_categorical(
                p_v_given_h[:, -self.n_labels :]
            )

        else:
            # finished
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (
            #  replace the pass and zeros below)
            support = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            # support[support < -75] = -75
            p_v_given_h = sigmoid(support)
            v = sample_binary(p_v_given_h)

        return p_v_given_h, v

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):
        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        p_h_given_v_dir = sigmoid(
            np.dot(visible_minibatch, self.weight_v_to_h) + self.bias_h
        )
        h = sample_binary(p_h_given_v_dir)

        return p_h_given_v_dir, h

    def get_v_given_h_dir(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:
            raise ValueError("This case should never be executed")

        else:
            p_v_given_h_dir = sigmoid(
                np.dot(hidden_minibatch, self.weight_h_to_v) + self.bias_v
            )
            s = sample_binary(p_v_given_h_dir)

        return p_v_given_h_dir, s
