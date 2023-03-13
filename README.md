## How to install
1. `pip install -r requirements.txt`
2. `cd custom-sklearn/`
3. `python setup.py install` - it should take a while, building a package from source.

The code from `test.py` is now runnable.

### Explanation:
custom-sklearn is scikit-learn==1.2.1 with custom `_multilayer_perceptron.py`
file to support additional solvers, our `custom_aadam`, 'custom_adamW' etc.
Note that each time you edit `custom-sklearn/sklearn/neural_network/_multilayer_perceptron.py` 
you have to build the package again in order to changes take place

# Abstract
Adam is a great ?solver?, it works way faster than SGD, and other solvers (?adagard rmsprop?).
And produce relatively good results. 
Unfortunately it fails to outperform SGD with Momentum in accuracy with enough training time.
But since Adam original revolutionary article at 2014, A lot has changed.
We wanted to compare some newer variants of Adam with the orinal version
to see how they hold in our mission to seek a even better solver.

# Project Mission


# Implementation
In order to save some precious coding time and help us to focus 
on our actual mission we rely on **scikit-learn** package to 

# Results
We tested the algorithms on two data sets:
- Cover Type
    - epochs - 40
    - train size - 100K


| Layers  | 
| ---  |
| Layers = 9  |
| ![](datasets_results_plots/Cover_Type_scores_ultra-deep_40epoches_100K.png )
| Layers = 6  |
| ![](datasets_results_plots/Cover_Type_scores_deep_40epoches_100K.png)
| Layers = 5  |
| ![](datasets_results_plots/Cover_Type_scores_Deep-fitted_40epoches_100K.png)  |
| Layers = 3  |
| ![](datasets_results_plots/Cover_Type_Regular_40epoches_100K.png)  |

- MNIST
    - epochs - 40
    - train size - 1K

| Layers  | 
| ---  |
| Layers = 9  |
| ![](datasets_results_plots/MNIST_digits_scores_ultra-deep_40epoch_1K.png )
| Layers = 6  |
| ![](datasets_results_plots/MNIST_digits_scores_deep_40epoch_1K.png)
| Layers = 5  |
| ![](datasets_results_plots/MNIST_digits_scores_deep_fitted_40epoch_1K.png)  |
| Layers = 3  |
| ![](datasets_results_plots/MNIST_digits_scores_regular_40epoch_1K.png)  |

# Conclusions

We can see the choice of the best optimizer will influenced mostly by the depth of the model (number of layers).
For example, Adam optimizer may work well for shallow models with fewer layers, while AdamW and AAdam optimizers may be better suited for deeper models with more layers.
In those experiments Adam optimizer performed better:
- MNIST digits: 
    - Layers = 3
    - Layers = 5
- Cover Type
    - Layers = 3
In those experiments AdamW and AAdam performed better:
- MNIST digits: 
    - Layers = 9
- Cover Type
    - Layers = 6
    - Layers = 9
   
This is because deeper models often have a larger number of parameters, which can make it more difficult for the optimizer to converge and avoid overfitting. Regularization techniques like weight decay (used in AdamW) and accumulated gradients (used in AAdam) can help address these issues, and may be more effective for deep models.

Therefore, if you have a deep neural network, it may be a good idea to consider using AdamW or AAdam optimizer instead of Adam optimizer. However, it is important to experiment with different optimizers and hyperparameters to find the best fit for your specific model and dataset.