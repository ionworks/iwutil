import numpy as np
import scipy.interpolate
from collections.abc import Callable
from numbers import Number


class Interpolator1D:
    """
    Base class for 1D interpolators with automatic data preprocessing.
    """

    __slots__ = ("_x", "_y", "_fill_value", "_interpolator")

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fill_value: Number | str | None = None,
        force_monotonic: bool | None = None,
    ):
        """
        Initialize the interpolator.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the data points. Must be 1D array.
        y : np.ndarray
            y-coordinates of the data points. Must have same length as x.
        fill_value : number or str, optional
            Value to use for out-of-bounds interpolation.
            Can be a number or "extrapolate". Default is np.nan.
        force_monotonic : bool, optional
            Whether to enforce monotonicity by removing non-increasing points.
            Default is False.
        """
        if fill_value is None:
            fill_value = np.nan
        self._check_fill_value(fill_value)

        if force_monotonic is None:
            force_monotonic = False

        x = np.array(x)
        y = np.array(y)
        for _vector in (x, y):
            if _vector.ndim != 1:
                raise ValueError("x and y must be 1D arrays")
        if x.size != y.size:
            raise ValueError("x and y must have the same length")
        if x.size < 2:
            raise ValueError("x and y must contain at least 2 elements")

        # Quick check to reverse the array if it is not strictly increasing
        # based on the end points
        if x[0] > x[-1]:
            x[:] = x[::-1]
            y[:] = y[::-1]

        monotonic_mask = self._get_monotonic_mask(x)
        is_non_monotonic = not monotonic_mask.all()
        if is_non_monotonic:
            if not force_monotonic:
                raise ValueError(
                    "x must be strictly monotonic if force_monotonic is False"
                )
            x = x[monotonic_mask]
            y = y[monotonic_mask]

        self._x = x
        self._y = y
        self._fill_value = fill_value
        self._interpolator = self._create_interpolator()

    def __call__(self, xi: np.ndarray | Number) -> np.ndarray:
        """
        Interpolate at the given x-coordinates.

        Parameters
        ----------
        xi : np.ndarray or number
            x-coordinates at which to interpolate. Can be scalar or array.

        Returns
        -------
        np.ndarray
            Interpolated values at xi.
        """
        xi = np.atleast_1d(xi)
        out = self.interpolator(xi)
        out = self._apply_fill_value(out, xi)
        return out

    def _apply_fill_value(self, out: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Apply the fill value to the output.

        Parameters
        ----------
        out : np.ndarray
            Output array.
        xi : np.ndarray
            x-coordinates at which to interpolate. Can be scalar or array.

        Returns
        -------
        np.ndarray
            Output array with fill value applied.
        """
        if self.extrapolate:
            return out

        x = self.x
        x_min, x_max = x.min(), x.max()
        if xi.min() < x_min or xi.max() > x_max:
            mask = (xi < x_min) | (xi > x_max)
            out[mask] = self.fill_value
        return out

    @staticmethod
    def _get_monotonic_mask(x: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask so that x[mask] is strictly increasing.

        Parameters
        ----------
        x : np.ndarray
            Input array to process.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates points to keep.
        """
        x = np.asarray(x)
        n = x.size

        # running maximum of x
        cummax = np.maximum.accumulate(x)

        # mask[0] is always True; afterwards True whenever cummax increases
        mask = np.empty(n, dtype=bool)
        mask[0] = True
        mask[1:] = cummax[1:] > cummax[:-1]

        return mask

    @property
    def x(self) -> np.ndarray:
        """
        The x-coordinates of the data points (sorted and cleaned).
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        The y-coordinates of the data points (corresponding to x).
        """
        return self._y

    @property
    def fill_value(self) -> Number | str:
        """
        Value used for out-of-bounds interpolation.
        """
        return self._fill_value

    @property
    def interpolator(self) -> Callable:
        """
        The underlying interpolation function.
        """
        return self._interpolator

    @property
    def extrapolate(self) -> bool:
        """
        Whether extrapolation is enabled.
        """
        return self.fill_value == "extrapolate"

    def derivative(self, order: int) -> Callable:
        """
        Compute the derivative of the interpolator.

        Parameters
        ----------
        order : int
            Order of the derivative.

        Returns
        -------
        callable
            Derivative function.
        """
        return self.interpolator.derivative(order)

    def antiderivative(self, order: int) -> Callable:
        """
        Compute the antiderivative of the interpolator.

        Parameters
        ----------
        order : int
            Order of the antiderivative.

        Returns
        -------
        callable
            Antiderivative function.
        """
        return self.interpolator.antiderivative(order)

    def _create_interpolator(self) -> Callable:
        """
        Create the underlying interpolator. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _create_interpolator")

    def _check_fill_value(self, fill_value: Number | str | None):
        """
        Validate the fill value.

        Parameters
        ----------
        fill_value : number or str, optional
            Fill value to validate.
        """
        if (
            fill_value is None
            or fill_value == "extrapolate"
            or isinstance(fill_value, Number)
        ):
            return

        raise ValueError("fill_value must be a number or 'extrapolate'")

    def __repr__(self) -> str:
        """
        String representation of the interpolator.
        """
        return f"{self.__class__.__name__}()"


class PchipInterpolator(Interpolator1D):
    """
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator.
    """

    def _create_interpolator(self) -> Callable:
        """
        Create a PCHIP interpolator.

        Returns
        -------
        callable
            PCHIP interpolator function.
        """
        return scipy.interpolate.PchipInterpolator(
            self.x, self.y, extrapolate=self.extrapolate
        )


class LinearInterpolator(Interpolator1D):
    """
    Linear spline interpolator.
    """

    def _create_interpolator(self) -> Callable:
        """
        Create a linear spline interpolator.

        Returns
        -------
        callable
            Linear spline interpolator function.
        """
        return scipy.interpolate.make_interp_spline(self.x, self.y, k=1)


class CubicSplineInterpolator(Interpolator1D):
    """
    Cubic spline interpolator.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fill_value: Number | str | None = None,
        force_monotonic: bool | None = None,
    ):
        """
        Initialize the cubic spline interpolator.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the data points. Must be 1D array.
        y : np.ndarray
            y-coordinates of the data points. Must have same length as x.
        fill_value : number or str, optional
            Value to use for out-of-bounds interpolation.
            Can be a number or "extrapolate". Default is np.nan.
        force_monotonic : bool, optional
            Whether to enforce monotonicity by removing non-increasing points.
            Default is False.
        """
        # Check if we have enough points for cubic spline (needs at least 4 points)
        if len(x) < 4:
            raise ValueError(
                f"Need at least 4 points for cubic spline interpolation, got {len(x)}"
            )

        super().__init__(x, y, fill_value, force_monotonic)

    def _create_interpolator(self) -> Callable:
        """
        Create a cubic spline interpolator.

        Returns
        -------
        callable
            Cubic spline interpolator function.
        """
        if self.extrapolate:
            bc_type = "natural"
        else:
            bc_type = None
        return scipy.interpolate.make_interp_spline(
            self.x, self.y, k=3, bc_type=bc_type
        )
