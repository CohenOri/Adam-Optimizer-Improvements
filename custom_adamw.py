import numpy as np
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer


class CustomAdamWOptimizer(AdamOptimizer):
    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weight_decay=1e-2):
        super().__init__(params, learning_rate_init, beta_1, beta_2, epsilon)
        self.weight_decay = weight_decay
        self.previous_update = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Update rules are of AAdam optimizer introduced at:
        https://arxiv.org/pdf/1711.05101v3.pdf
        or elaborated at:
        https://towardsdatascience.com/why-adamw-matters-736223f31b5d
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
        self.ms = [
            self.beta_1 * m + (1 - self.beta_1) * grad
            for m, grad in zip(self.ms, grads)
        ]
        self.vs = [
            self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            for v, grad in zip(self.vs, grads)
        ]
        self.learning_rate = (
                self.learning_rate_init
                * np.sqrt(1 - self.beta_2**self.t)
                / (1 - self.beta_1**self.t)
        )
        updates = [
            -self.learning_rate * (m / (np.sqrt(v) + self.epsilon) + self.weight_decay*prev)
            for m, v, prev in zip(self.ms, self.vs, self.previous_update)
        ]
        return updates
