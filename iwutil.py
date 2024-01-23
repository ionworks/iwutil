import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def subplots_autolayout(
    n, *args, n_rows=None, figsize=None, layout="constrained", **kwargs
):
    """
    Create a subplot element with a
    """
    n_rows = n_rows or int(n // np.sqrt(n))
    n_cols = int(np.ceil(n / n_rows))

    figwidth_default = min(15, 4 * n_cols)
    figheight_default = min(8, 1 + 3 * n_rows)
    figsize = figsize or (figwidth_default, figheight_default)
    fig, axes = plt.subplots(
        n_rows, n_cols, *args, figsize=figsize, layout=layout, **kwargs
    )
    # if we just have a single axis, make sure we are returning an array instead
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    return fig, axes


def process_folder_file(folder, file):
    """
    Process folder and file to create a full path. If folder does not exist, create it.

    Parameters
    ----------
    folder : str
        Folder to save file in
    file : str
        File name

    Returns
    -------
    str
        Full path to file
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, file)


def save_json(params, folder, file):
    """
    Save params to a json file in folder/file

    Parameters
    ----------
    params : dict
        Dictionary of parameters
    folder : str
        Folder to save file in
    file : str
        File name
    """
    full_name = process_folder_file(folder, file)
    with open(full_name, "w") as f:
        json.dump(params, f, indent=2)


def save_csv(df, folder, file):
    """
    Save df to a csv file in folder/file

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    folder : str
        Folder to save file in
    file : str
        File name
    """
    full_name = process_folder_file(folder, file)
    df.to_csv(full_name, index=False)


def save_fig(fig, folder, file):
    """
    Save fig to a file in folder

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    folder : str
        Folder to save file in
    file : str
        File name
    """
    full_name = process_folder_file(folder, file)
    fig.savefig(full_name)
