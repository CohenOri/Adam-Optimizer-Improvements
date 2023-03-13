import numpy as np
from numpy import absolute
from numpy import sign
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer


class CustomAAdamOptimizer(AdamOptimizer):
    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate_init, beta_1, beta_2, epsilon)
        self.previous_update = [np.zeros_like(param) for param in params]
        self.previous_previous_update = [np.zeros_like(param) for param in params]
        self.ds = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Update rules are of AAdam optimizer introduced at:

        and slightly adjusted for sklearn update rules

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        self.ds = [
            absolute(prev - prev_prev) * sign(grad) * (1 - self.beta_1)
            for prev, prev_prev, grad in zip(self.previous_update, self.previous_previous_update, grads)]
        updates = [-self.learning_rate * m * self.beta_1 / (np.sqrt(v + self.epsilon)) + d
                   for m, v, d in zip(self.ms, self.vs, self.ds)]
        if np.all((self.previous_previous_update == 0)) and np.all((self.previous_update == 0)):
            self.previous_previous_update = updates
            self.previous_update = updates
        else:
            self.previous_previous_update = self.previous_update
            self.previous_update = updates
        return updates
