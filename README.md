## How to run
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
Adam is a great Optimizer (often called solver), 
Introduced in 2014 - [Adam: A method for
stochastic optimization]() and still one of the most popular optimizers.
It converges way faster than SGD, and other solvers (?adagard rmsprop?).
meanwhile, producing relatively good results.
Some claim that with enough training time SGD with Momentum might be able to 
outperform Adam in accuracy.
Adam original article has been published in 2014, since then a few improvements has been published.
In this workshop we implemented two of them:
1. AAdam (demonstrated [here](https://openreview.net/pdf?id=HJfpZq1DM))
2. AdamW - very popular in the field of deep neural networks for NLP (demonstrated [here](https://openreview.net/pdf?id=HJfpZq1DM))

Then we conducted accuracy per epoch comparison and plotted the results.
Our results support the claims that the "improved" versions of Adam 
indeed preform better (with almost no exceptions) in real life scenarios. 
In our experience we received slightly better accuracy, 
or faster training time with the same accuracy, therefore we suggest using them.

# Experience Setup
Experience setup is as following:
### Optimizers:
We compared 4 different optimizers
1. Adam - built-in sklearn
2. AAdam - Implemented manually in `custom_aadam.py`
3. AdamW - Implemented manually in `custom_adamw.py`
4. SGD - built-in sklearn

### Datasets:
Over 2 different datasets:
1. MNIST Digits
2. Cover Type
    
Then we coundcted a comprasion of 4 diffrent optimizers on 4 diffrent Neural Networks Aritcheus over two diffrent Datasets
and conducted a trough comparsion in
some of them and compare them with the orignal Adam to verify  
We intendtend to test the   
But since Adam original revolutionary article at 2014, A lot has changed.
We wanted to compare some newer variants of Adam with the orinal version
to see how they hold in our mission to seek a even better solver.

# Project Mission


# Implementation
In order to save some precious coding time and help us to focus 
on our actual mission we rely on **scikit-learn** package to 

# Results

# Conclusions
