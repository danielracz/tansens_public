import json

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from scipy.stats import pearsonr as pr

def get_class_gaps(class_losses_json_path):
    """ Binary classes for each output neuron had been calculated
        retrospectively, therefore it is loaded from a different json file.

    Args:
        class_losses_json_path (_type_): Absolute path of json.

    Returns:
        _type_: numpy array of generalization gaps for each class
    """
    with open(class_losses_json_path, 'r') as f:
        data = json.load(f)
        class_losses_train = np.array(data["train_losses"])
        class_losses_test = np.array(data["test_losses"])
    class_gaps = np.abs(class_losses_train - class_losses_test)
    return class_gaps

def plot_tansens(dataset, json_path, class_gaps=None, show=False):
    """ Generates the plot of tangent sensitivities and generalization gaps.

    Args:
        dataset (_type_): Either cifar10 or mnist.
        json_path (_type_): Path of metrics generated during training.
        class_gaps (_type_, optional): Array of class gaps calculated
                                       by get_class_gaps. Defaults to None.
        show (bool, optional): If true, plt.show() is invoked.
                               Defaults to False.

    Returns:
        _type_: Pyplot figure.
    """
    assert dataset in ["cifar10", "mnist"]
    if dataset == "cifar10":
        test_data= torchvision.datasets.CIFAR10("./", train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
        counts = torch.unique(torch.Tensor(test_data.targets),
                              return_counts=True)[1]
    else: # mnist
        test_data = torchvision.datasets.MNIST("./", train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
        counts = torch.unique(test_data.test_labels, return_counts=True)[1]
    
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    train_loss = np.array(metrics["train_loss"])
    test_loss = np.array(metrics["test_loss"])
    gap = np.abs(train_loss - test_loss)

    num_epochs = len(train_loss)

    tansens = torch.Tensor(metrics["tansens"])
    avg_tansenses = torch.bmm(tansens,
                              counts.unsqueeze(1).repeat(num_epochs,
                                                         1,
                                                         1).float()).squeeze()
    
    if class_gaps:
        fig, axs = plt.subplots(3, 1, figsize=(8, 13), dpi=300)
        axs[0].plot(range(num_epochs), train_loss,
                    alpha=0.5, label='Training loss')
        axs[0].plot(range(num_epochs), test_loss,
                    alpha=0.5, label='Test_loss')
        axs[0].plot(range(num_epochs), gap,
                    alpha=0.5, label='Empirical generalization gap')
        axs[0].plot(range(num_epochs),
                    np.mean(avg_tansenses.numpy()/ 10000000,
                            axis=1),alpha=0.5,
                    label="Mean tangent sens. (scaled)")
        axs[0].set_xlabel("Epochs")
        axs[0].legend()

        for cls in range(10):
            axs[1].plot(range(60), class_gaps[:, cls],
                        alpha=0.5, label=f"Class gaps of output neuron {cls}")
        axs[1].legend(fontsize=8)

        for cls in range(10):
            corr = pr(avg_tansenses[:, cls].squeeze(), gap).statistic
            corr_class = pr(avg_tansenses[:, cls].squeeze(),
                            class_gaps[:, cls]).statistic
            axs[2].plot(range(60), avg_tansenses[:, cls] / 10_000,
                        alpha=0.5,
                        label=f"Tangent sens. of output neuron {cls}\n corr.: {corr:.2f}, corr with class gap: {corr_class:.2f}")
        axs[2].legend(fontsize=8)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
        axs[0].plot(range(num_epochs), train_loss, alpha=0.5,
                    label='Training loss')
        axs[0].plot(range(num_epochs), test_loss, alpha=0.5,
                    label='Test loss')
        axs[0].plot(range(num_epochs), gap, alpha=0.5,
                    label='Empirical generalization gap')
        axs[0].plot(range(num_epochs),
                    np.mean(avg_tansenses.numpy() / 10000000,axis=1),
                    alpha=0.5, label="Mean tangent sens. (scaled)")
        axs[0].set_xlabel("Epochs")
        axs[0].legend()

        num_classes = counts.shape[0]
        for cls in range(num_classes):
            corr = pr(avg_tansenses[:, cls].squeeze(), gap)[0]
            axs[1].plot(range(60), avg_tansenses[:, cls] / 10_000,
                        alpha=0.5,
                        label=f"Tangent sens. of output neuron {cls}\n corr.: {corr:.2f}")
        axs[1].set_xlabel("Epochs")
        axs[1].legend(fontsize=8)

    if show:
        plt.show()
    return fig
