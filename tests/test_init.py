import iwutil


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
