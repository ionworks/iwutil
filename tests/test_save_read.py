import iwutil
import pandas as pd
import tempfile
import pytest
import pathlib


@pytest.mark.parametrize("file_format", ["df", "csv", "parquet", "json", "txt"])
@pytest.mark.parametrize("path_format", ["posix path", "str"])
def test_save_read_df(file_format, path_format):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    df_read = pd.DataFrame()
    if file_format == "df":
        df_read = iwutil.read_df(df)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            name = "test." + file_format

            if path_format == "posix path":
                temp_dir = pathlib.Path(temp_dir)
                file = temp_dir / name
            else:
                file = temp_dir + "/" + name

            if file_format == "csv":
                iwutil.save.csv(df, file)
            elif file_format == "parquet":
                iwutil.save.parquet(df, file)
            elif file_format == "json":
                iwutil.save.json(df.to_dict(orient="list"), file)
            elif file_format == "txt":
                iwutil.save.txt(df, file)
            else:
                raise NotImplementedError(f"Test does not cover format: {file_format}")

            df_read = iwutil.read_df(file)
    assert df.equals(df_read)


def test_read_df_kwargs():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file = temp_dir + "/test.csv"
        iwutil.save.csv(df, file)

        df_read = iwutil.read_df(file, usecols=["a"])
        assert df_read.equals(pd.DataFrame({"a": [1, 2, 3]}))


def test_read_json():
    data = {"a": 1, "b": 4}
    with tempfile.TemporaryDirectory() as temp_dir:
        file = temp_dir + "/test.json"
        iwutil.save.json(data, file)
        assert iwutil.read_json(file) == data
