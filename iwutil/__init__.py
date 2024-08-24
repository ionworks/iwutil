from . import save

import matplotlib.pyplot as plt
import numpy as np
from functools import singledispatch
import pandas as pd
from pathlib import Path


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


def check_and_combine_options(default_options, custom_options=None):
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


@singledispatch
def read_df(file):
    raise NotImplementedError(f"Reading type {type(file)} not implemented")


@read_df.register
def _(file: str | Path):
    file_extension = Path(file).suffix[1:]

    if file_extension == "csv":
        return pd.read_csv(file)
    elif file_extension in ["xls", "xlsx"]:
        return pd.read_excel(file)
    elif file_extension == "json":
        return pd.read_json(file)
    elif file_extension == "parquet":
        return pd.read_parquet(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


@read_df.register
def _(file: pd.DataFrame):
    return file
