import iwutil
import numpy as np
import pytest


@pytest.fixture
def quadratic_data():
    """Standard quadratic data for testing: y = x^2."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])
    return x, y


@pytest.fixture
def non_monotonic_data():
    """Non-monotonic data for testing monotonicity handling."""
    x = np.array([0, 1, 0.5, 2, 3])
    y = np.array([0, 1, 0.5, 4, 9])
    return x, y


@pytest.fixture
def reverse_data():
    """Reverse-ordered data for testing automatic reordering."""
    x = np.array([4, 3, 2, 1, 0])
    y = np.array([16, 9, 4, 1, 0])
    return x, y


@pytest.fixture
def minimal_data():
    """Minimal data for testing edge cases."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    return x, y


@pytest.fixture
def test_points():
    """Standard test points for interpolation."""
    return np.array([0.5, 1.5, 2.5])


def test_interpolator1d_basic(quadratic_data, test_points):
    # Test basic interpolation with simple data
    x, y = quadratic_data

    interp = iwutil.interpolate.PchipInterpolator(x, y)
    result = interp(test_points)

    assert len(result) == 3
    assert all(np.isfinite(result))
    assert result[0] > 0 and result[0] < 1  # Should be between 0 and 1
    assert result[1] > 1 and result[1] < 4  # Should be between 1 and 4
    assert result[2] > 4 and result[2] < 9  # Should be between 4 and 9


@pytest.mark.parametrize(
    "interpolator_class",
    [
        iwutil.interpolate.PchipInterpolator,
        iwutil.interpolate.LinearInterpolator,
        iwutil.interpolate.CubicSplineInterpolator,
    ],
)
def test_interpolator1d_different_methods(quadratic_data, interpolator_class):
    # Test different interpolation methods
    x, y = quadratic_data

    # Skip cubic spline for insufficient points
    if interpolator_class == iwutil.interpolate.CubicSplineInterpolator and len(x) < 4:
        pytest.skip("Cubic spline needs at least 4 points")

    interp = interpolator_class(x, y)
    result = interp(1.5)

    # Should be finite and reasonable
    assert np.isfinite(result)
    assert result > 1 and result < 4


def test_interpolator1d_extrapolation(quadratic_data):
    # Test extrapolation behavior
    x, y = quadratic_data

    # Test with fill_value (default behavior)
    interp_fill = iwutil.interpolate.PchipInterpolator(x, y, fill_value=np.nan)
    result_fill = interp_fill(x.max() + 1)  # Outside range
    assert np.isnan(result_fill)

    # Test with extrapolation
    interp_extrap = iwutil.interpolate.PchipInterpolator(x, y, fill_value="extrapolate")
    result_extrap = interp_extrap(x.max() + 1)  # Outside range
    assert np.isfinite(result_extrap)
    assert result_extrap > y.max()  # Should extrapolate beyond the last point


def test_interpolator1d_monotonic_handling(non_monotonic_data):
    # Test handling of non-monotonic data
    x, y = non_monotonic_data

    # Should fail with force_monotonic=False (default)
    with pytest.raises(ValueError):
        iwutil.interpolate.PchipInterpolator(x, y, force_monotonic=False)

    # Should work with force_monotonic=True
    interp = iwutil.interpolate.PchipInterpolator(x, y, force_monotonic=True)
    result = interp(1.5)
    assert np.isfinite(result)


def test_interpolator1d_reverse_order(reverse_data):
    # Test handling of reverse-ordered data
    x, y = reverse_data

    interp = iwutil.interpolate.PchipInterpolator(x, y)
    result = interp(1.5)
    assert np.isfinite(result)
    assert result > 1 and result < 4


def test_interpolator1d_insufficient_points(minimal_data):
    # Test error handling for insufficient points
    x, y = minimal_data

    # Should work for linear interpolation
    interp_linear = iwutil.interpolate.LinearInterpolator(x, y)
    result = interp_linear(0.5)
    assert np.isfinite(result)

    # Should fail for cubic spline (needs at least 4 points)
    with pytest.raises(ValueError):
        iwutil.interpolate.CubicSplineInterpolator(x, y)


@pytest.mark.parametrize(
    "interpolator_class",
    [
        iwutil.interpolate.PchipInterpolator,
        iwutil.interpolate.LinearInterpolator,
        iwutil.interpolate.CubicSplineInterpolator,
    ],
)
def test_interpolator1d_properties(quadratic_data, interpolator_class):
    # Test that properties work correctly
    x, y = quadratic_data

    # Skip cubic spline for insufficient points
    if interpolator_class == iwutil.interpolate.CubicSplineInterpolator and len(x) < 4:
        pytest.skip("Cubic spline needs at least 4 points")

    interp = interpolator_class(x, y)

    assert np.array_equal(interp.x, x)
    assert np.array_equal(interp.y, y)
    assert np.isnan(interp.fill_value)
    assert callable(interp.interpolator)
    assert not interp.extrapolate


def test_interpolator1d_invalid_inputs(quadratic_data):
    # Test error handling for invalid inputs
    x, y = quadratic_data

    # Test invalid fill_value
    with pytest.raises(ValueError):
        iwutil.interpolate.PchipInterpolator(x, y, fill_value="invalid")

    # Test mismatched lengths
    with pytest.raises(ValueError):
        iwutil.interpolate.PchipInterpolator(x, y[:-1])

    # Test insufficient points
    with pytest.raises(ValueError):
        iwutil.interpolate.PchipInterpolator(x[:1], y[:1])

    # Test 2D arrays
    with pytest.raises(ValueError):
        iwutil.interpolate.PchipInterpolator(x.reshape(-1, 1), y)


@pytest.mark.parametrize(
    "interpolator_class",
    [
        iwutil.interpolate.PchipInterpolator,
        iwutil.interpolate.LinearInterpolator,
        iwutil.interpolate.CubicSplineInterpolator,
    ],
)
def test_interpolator1d_repr(quadratic_data, interpolator_class):
    # Test string representation
    x, y = quadratic_data

    # Skip cubic spline for insufficient points
    if interpolator_class == iwutil.interpolate.CubicSplineInterpolator and len(x) < 4:
        pytest.skip("Cubic spline needs at least 4 points")

    interp = interpolator_class(x, y)
    repr_str = repr(interp)

    assert interpolator_class.__name__ in repr_str


@pytest.mark.parametrize(
    "interpolator_class",
    [
        iwutil.interpolate.PchipInterpolator,
        iwutil.interpolate.LinearInterpolator,
        iwutil.interpolate.CubicSplineInterpolator,
    ],
)
def test_method_consistency(quadratic_data, interpolator_class):
    """Test that each method produces consistent results."""
    x, y = quadratic_data
    test_point = 1.5

    # Skip cubic spline for insufficient points
    if interpolator_class == iwutil.interpolate.CubicSplineInterpolator and len(x) < 4:
        pytest.skip("Cubic spline needs at least 4 points")

    interp = interpolator_class(x, y)
    result = interp(test_point)

    # Result should be finite and reasonable
    assert np.isfinite(result), (
        f"{interpolator_class.__name__} produced non-finite result"
    )
    assert result > 1 and result < 4, (
        f"{interpolator_class.__name__} result out of expected range"
    )


@pytest.mark.parametrize(
    "interpolator_class",
    [
        iwutil.interpolate.PchipInterpolator,
        iwutil.interpolate.LinearInterpolator,
        iwutil.interpolate.CubicSplineInterpolator,
    ],
)
def test_derivative(quadratic_data, interpolator_class):
    """Test the derivative method."""
    x, y = quadratic_data

    # Skip cubic spline for insufficient points
    if interpolator_class == iwutil.interpolate.CubicSplineInterpolator and len(x) < 4:
        pytest.skip("Cubic spline needs at least 4 points")

    interp = interpolator_class(x, y)
    test_point = 1.5

    # Test derivative - returns a function that can be evaluated
    derivative_func = interp.derivative(1)
    derivative = derivative_func(test_point)

    assert isinstance(derivative, np.ndarray)
    assert np.isfinite(derivative)
    # For y = x^2, derivative should be 2x, so at x=1.5, derivative should be ~3
    assert derivative > 2 and derivative < 4


@pytest.mark.parametrize(
    "interpolator_class",
    [
        iwutil.interpolate.PchipInterpolator,
        iwutil.interpolate.LinearInterpolator,
        iwutil.interpolate.CubicSplineInterpolator,
    ],
)
def test_antiderivative(quadratic_data, interpolator_class):
    """Test the antiderivative method."""
    x, y = quadratic_data

    # Skip cubic spline for insufficient points
    if interpolator_class == iwutil.interpolate.CubicSplineInterpolator and len(x) < 4:
        pytest.skip("Cubic spline needs at least 4 points")

    interp = interpolator_class(x, y)
    test_point = 1.5

    # Test antiderivative - returns a function that can be evaluated
    antiderivative_func = interp.antiderivative(1)
    antiderivative = antiderivative_func(test_point)

    assert isinstance(antiderivative, np.ndarray)
    assert np.isfinite(antiderivative)
    # For y = x^2, antiderivative should be x^3/3, so at x=1.5, should be ~1.125
    assert antiderivative > 0.5 and antiderivative < 2.0
