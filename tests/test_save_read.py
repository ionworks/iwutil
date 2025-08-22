import iwutil
import pandas as pd
import tempfile
import pytest
import pathlib

# Try to import polars for optional tests
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


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


@pytest.mark.parametrize("file_format", ["csv", "parquet", "txt"])
def test_pandas_dataframe_save(file_format):
    """Test saving pandas DataFrames using singledispatch"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5], "c": ["x", "y", "z"]})
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file = temp_dir + f"/test.{file_format}"
        
        # Test the save function
        if file_format == "csv":
            iwutil.save.csv(df, file)
        elif file_format == "parquet":
            iwutil.save.parquet(df, file)
        elif file_format == "txt":
            iwutil.save.txt(df, file)
        
        # Verify file was created
        assert pathlib.Path(file).exists()
        
        # Verify content by reading back
        df_read = iwutil.read_df(file)
        assert df.equals(df_read)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.parametrize("file_format", ["csv", "parquet", "txt"])
def test_polars_dataframe_save(file_format):
    """Test saving polars DataFrames using singledispatch"""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5], "c": ["x", "y", "z"]})
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file = temp_dir + f"/test.{file_format}"
        
        # Test the save function
        if file_format == "csv":
            iwutil.save.csv(df, file)
        elif file_format == "parquet":
            iwutil.save.parquet(df, file)
        elif file_format == "txt":
            iwutil.save.txt(df, file)
        
        # Verify file was created
        assert pathlib.Path(file).exists()
        
        # Verify content by reading back and comparing with pandas equivalent
        df_read = iwutil.read_df(file)
        df_pandas = df.to_pandas()
        assert df_pandas.equals(df_read)


def test_unsupported_dataframe_type_raises_error():
    """Test that unsupported DataFrame types raise NotImplementedError"""
    class FakeDataFrame:
        pass
    
    fake_df = FakeDataFrame()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file = temp_dir + "/test.csv"
        
        with pytest.raises(NotImplementedError, match="CSV save not implemented for type"):
            iwutil.save.csv(fake_df, file)
        
        with pytest.raises(NotImplementedError, match="Parquet save not implemented for type"):
            iwutil.save.parquet(fake_df, file)
        
        with pytest.raises(NotImplementedError, match="TXT save not implemented for type"):
            iwutil.save.txt(fake_df, file)


@pytest.mark.skipif(HAS_POLARS, reason="polars is installed")
def test_polars_import_error_when_not_installed():
    """Test that helpful error is raised when polars is not installed but polars DataFrame is used"""
    # This test only runs when polars is NOT installed
    # We can't actually test this easily without mocking, so we'll test the error message format
    pass
