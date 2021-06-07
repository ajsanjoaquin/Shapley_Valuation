# Equitable Valuation of Data Using Shapley Values (PyTorch Implementation)

This is a PyTorch reimplementation of Computing Shapley Values via Truncated Monte Carlo sampling from 
[What is your data worth? Equitable Valuation of Data](https://arxiv.org/abs/1904.02868) by Amirata Ghorbani and James Zou.
The original implementation (In Tensorflow) can be found [here](https://github.com/amiratag/DataShapley).

This implementation is currently designed for neural networks, and the only available performance metric is model classification accuracy, 
but contributions to expand the implementation are welcome. 

- [Why Compute Shapley Values?](#why-compute-shapley-values?)
- [Requirements](#requirements)
- [Usage](#usage)

## Why Compute Shapley Values?

Computing Shapley values help when you need a system to rank the importance of your training data, 
which may arise when you need to prune your training data of harmful images, 
or when you need to provide compensation for data from multiple sources.

It differs from computing the value based on the leave-one-out method (LOO), 
because Shapley values satisfy three main properties:

1. Null Data: If a datum does not change the model performance if it is added to any subset of the training data, then its value is zero.
2. Equality: For any data x & y, if x has equal contribution to y when added to any subset of the training data, then x and y have the same Shapley value. 
3. Additive: If datum x contributes S_x(d_1) and S_x(d_2) to test data 1 and 2, respectively, then the value of x for both points is S_x(d_1) + S_x(d_2).

## Requirements

* Python 3.6 or later
* PyTorch 1.0 or later
* NumPy 1.12 or later
* Pickle
* Tqdm

## Usage

```python
from tmc import DShap

# Supplied by the user:
model = get_my_model()
train_set, test_set = get_my_datasets()

dshap = DShap(model, train_set, testset, directory='your_directory')

dshap.run(save_every=100, err=0.1, tolerance=0.01)
```

This outputs a pickle file that contains the sampled Shapley Values. You can convert this into a numpy array of dimensions (Iterations x # of Training Points).


## LICENSE
<a rel="license" href="https://opensource.org/licenses/MIT"><img alt="Creative Commons Licence" style="border-width:0" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/MIT_logo.svg/220px-MIT_logo.svg.png" /></a>
