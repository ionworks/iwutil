import os
import json
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


def check_and_combine_options(self, default_options, custom_options=None):
    """
    Check that all required options are provided, and combine default and custom options

    Parameters
    ----------
    default_options : dict
        Dictionary of default options
    custom_options : dict, optional
        Dictionary of custom options, by default None

    Returns
    -------
    dict
        Combined options
    """

    if custom_options is None:
        custom_options = {}

    # Check that all custom option keys have a default
    for k in custom_options:
        if k not in default_options:
            raise ValueError(f"Option '{k}' not recognized")

    # If any default options are marked as "[required]", check that they are provided
    for k, v in default_options.items():
        if v == "[required]" and k not in custom_options:
            raise ValueError(f"Option '{k}' is required")

    return {**default_options, **custom_options}
