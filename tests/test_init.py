import iwutil
import tempfile
from pathlib import Path


def test_check_and_combine_options():
    # Test case 1: Default options only
    default_options = {"a": 1, "b": 2}
    result = iwutil.check_and_combine_options(default_options)
    assert result == {"a": 1, "b": 2}

    # Test case 2: Custom options
    default_options = {"a": 1, "b": 2}
    custom_options = {"b": 3}
    result = iwutil.check_and_combine_options(default_options, custom_options)
    assert result == {"a": 1, "b": 3}

    # Test case 3: Required option
    default_options = {"a": 1, "b": "[required]"}
    custom_options = {"b": 2}
    result = iwutil.check_and_combine_options(default_options, custom_options)
    assert result == {"a": 1, "b": 2}

    # Test case 4: Missing required option
    default_options = {"a": 1, "b": "[required]"}
    custom_options = {"a": 2}
    try:
        iwutil.check_and_combine_options(default_options, custom_options)
    except ValueError as e:
        assert str(e) == "Option 'b' is required"

    # Test case 5: Unrecognized option
    default_options = {"a": 1, "b": 2}
    custom_options = {"c": 3}
    try:
        iwutil.check_and_combine_options(default_options, custom_options)
    except ValueError as e:
        assert str(e) == "Option 'c' not recognized"


def test_copyfile():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        # Create a source file
        src_file = tempdir_path / "source.txt"
        src_file.write_text("Hello, World!")

        # Define a destination file in a non-existent directory
        dst_file = tempdir_path / "subdir" / "destination.txt"

        # Copy the file
        iwutil.copyfile(src_file, dst_file)

        # Check if the destination file exists and has the correct content
        assert dst_file.exists()
        assert dst_file.read_text() == "Hello, World!"

        # Check if the parent directory was created
        assert dst_file.parent.is_dir()

        # Test overwriting an existing file
        new_content = "New content"
        src_file.write_text(new_content)
        iwutil.copyfile(src_file, dst_file)
        assert dst_file.read_text() == new_content

        # Test copying to an existing directory
        existing_dir = tempdir_path / "existing_dir"
        existing_dir.mkdir()
        dst_file2 = existing_dir / "destination2.txt"
        iwutil.copyfile(src_file, dst_file2)
        assert dst_file2.exists()
        assert dst_file2.read_text() == new_content


def test_this_dir():
    # Test with a file path as a string
    file_path = "/path/to/some/file.txt"
    expected_dir = Path("/path/to/some")
    assert iwutil.this_dir(file_path) == expected_dir

    # Test with a file path as a Path object
    file_path = Path("/another/path/to/file.py")
    expected_dir = Path("/another/path/to")
    assert iwutil.this_dir(file_path) == expected_dir

    # Test with a relative path
    file_path = "relative/path/to/file.csv"
    expected_dir = Path("relative/path/to")
    assert iwutil.this_dir(file_path) == expected_dir

    # Test with just a filename (should return current directory)
    file_path = "filename.txt"
    expected_dir = Path(".")
    assert iwutil.this_dir(file_path) == expected_dir

    # Test with empty string (should return current directory)
    file_path = ""
    expected_dir = Path(".")
    assert iwutil.this_dir(file_path) == expected_dir
