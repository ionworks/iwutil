from pathlib import Path
from json import dump


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
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path / file


def json(params, folder, file):
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
        dump(params, f, indent=2)


def csv(df, folder, file):
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


def parquet(df, folder, file):
    """
    Save df to a parquet file in folder/file

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
    df.to_parquet(full_name)


def fig(fig, folder, file):
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
