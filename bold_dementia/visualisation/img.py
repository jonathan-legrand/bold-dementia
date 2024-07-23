import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns


def show_slices(slices):
    
    """ Function to display column of image slices """
    
    names = ("saggital", "coronal", "axial")
    fig, axes = plt.subplots(len(slices), figsize=(10, 10))
    
    for i, slice in enumerate(slices):
    
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].set_title(names[i])
        
    return fig


def make_and_show_middle_slices(volume):
    """
    Stupid name
    """
    if volume.ndim == 4:
        volume = volume.mean(axis=3)
    h, w, d = volume.shape
    return show_slices(
        (
            volume[h//2, :, :],
            volume[:, w//2, :],
            volume[:, :, d//2]
        )
    )