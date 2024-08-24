import iwutil
import pandas as pd
import tempfile
import pytest
import pathlib

@pytest.mark.parametrize("file_format", ["df", "csv", "parquet", "json", "csv"])
@pytest.mark.parametrize("path_format", ["posix path", "str"])
def test_save_read_df(file_format, path_format):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    if file_format == "df":
        df_read = iwutil.read_df(df)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            name = "test." + file_format
            
            if path_format == "posix path":
                temp_dir = pathlib.Path(temp_dir)
                temp_file_path = temp_dir / name
            else:
                temp_file_path = temp_dir + "/" + name
            
            
            if file_format == "csv":
                iwutil.save.csv(df, temp_dir, name)
            elif file_format == "parquet":
                iwutil.save.parquet(df, temp_dir, name)
            elif file_format == "json":
                iwutil.save.json(df.to_dict(orient="list"), temp_dir, name)

            df_read = iwutil.read_df(temp_file_path)
    assert df.equals(df_read)
