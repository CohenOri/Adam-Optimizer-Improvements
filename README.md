# How to install
1. `pip install -r requirements.txt`
2. `cd custom-sklearn/`
3. `python setup.py install` - it should take a while, building a package from source.

The code can be run now.

### Explanation:
custom-sklearn is scikit-learn==1.2.1 with custom `_multilayer_perceptron.py`
file to support additional solvers, our `custom_aadam`, 'custom_adamW' etc.
Note that each time you edit `custom-sklearn/sklearn/neural_network/_multilayer_perceptron.py` 
you have to build the package again in order to changes take place 
