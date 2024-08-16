import iwutil
import pandas as pd
import tempfile
import pytest


@pytest.mark.parametrize("file_format", ["df", "csv", "parquet", "json"])
def test_save_read_df(file_format):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    if file_format == "df":
        df_read = iwutil.read_df(df)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            name = "test." + file_format
            temp_file = temp_dir + "/" + name
            if file_format == "csv":
                iwutil.save.csv(df, temp_dir, name)
            elif file_format == "parquet":
                iwutil.save.parquet(df, temp_dir, name)
            elif file_format == "json":
                iwutil.save.json(df.to_dict(orient="list"), temp_dir, name)

            df_read = iwutil.read_df(temp_file)
    assert df.equals(df_read)
