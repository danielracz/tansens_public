# tansens_public

This repo contains the Python code of the experiments presented in the paper
"Optimization dependent generalization bound for ReLU networks based on sensitivity in the tangent bundle"
submitted to TBA.

## Requirements

See the file [jax.yml](jax.yml) exported from an [Anaconda](https://www.anaconda.com/download#downloads) environment.

## Usage

Run [train.py](train.py) to train the selected Deep ReLU network. The code calculates the mean of Tangent Sensitivity norms for each output neuron and class. The result is saved under the [metrics](metrics) folder in a JSON format while the parameters are saved in the [params](params) folder for each epoch. In order to plot the generalization gaps for each output neuron, run [calc_class_losses.py](calc_class_losses.py) after the training (or include this code in the training loop).

Use the functions in [plot.py](plot.py) to plot the results in the same format as they are presented in the paper.
