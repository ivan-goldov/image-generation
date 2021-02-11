import torchvision
import matplotlib.pyplot as plt
import numpy as np


def tensor2image(tensor, mean=None, std=None):
    tensor = tensor.detach()
    tensor = torchvision.utils.make_grid(tensor).to('cpu')
    tensor = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5]) 
    tensor = std * tensor + mean
    return np.clip(tensor, 0, 1)


def plot(*tensors):
    size = len(tensors)
    if size == 1:
        plt.imshow(tensor2image(tensors[0]))
    else:
        fig, axes= plt.subplots(nrows=1, ncols=size, figsize=(size * 4, 4))   
        for ax, t in zip(axes, tensors):
            ax.imshow(tensor2image(t))
        plt.show()