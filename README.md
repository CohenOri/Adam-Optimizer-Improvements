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

# Conclusions
