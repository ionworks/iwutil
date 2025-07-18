import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline
from collections.abc import Callable
from numbers import Number


class Interpolator1D:
    """
    A 1D interpolator with automatic data preprocessing and multiple methods.
    """

    __slots__ = ("_x", "_y", "_method", "_fill_value", "_interpolator")

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str | None = None,
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
        method : str, optional
            Interpolation method. Options: "pchip" (default), "linear", "cubic spline".
        fill_value : number or str, optional
            Value to use for out-of-bounds interpolation.
            Can be a number or "extrapolate". Default is np.nan.
        force_monotonic : bool, optional
            Whether to enforce monotonicity by removing non-increasing points.
            Default is False.
        """
        if method is None:
            method = "pchip"
        self._check_method(method)

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
        self._method = method
        self._fill_value = fill_value
        self._interpolator = self._get_interpolator(method)

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
        # Convert scalar input to array for processing
        out = self.interpolator(xi)
        if self.extrapolate:
            return out

        x = self.x
        mask = (xi < x.min()) | (xi > x.max())
        # Handle both scalar and array masks
        if mask.any():
            out[mask] = self.fill_value
        return out

    def _get_default_methods(self) -> list[str]:
        """
        Get the list of available interpolation methods.
        """
        return ["pchip", "linear", "cubic spline"]

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
    def method(self) -> str:
        """
        The interpolation method being used.
        """
        return self._method

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

    def _get_pchip_interpolator(self, x: np.ndarray, y: np.ndarray) -> Callable:
        """
        Create a PCHIP interpolator.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the data points.
        y : np.ndarray
            y-coordinates of the data points.

        Returns
        -------
        callable
            PCHIP interpolator function.
        """
        if self.extrapolate:
            extrapolate = True
        else:
            extrapolate = False
        return PchipInterpolator(x, y, extrapolate=extrapolate)

    def _get_spline_interpolator(
        self, x: np.ndarray, y: np.ndarray, k: int
    ) -> Callable:
        """
        Create a spline interpolator with specified degree.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the data points.
        y : np.ndarray
            y-coordinates of the data points.
        k : int
            Degree of the spline (1 for linear, 3 for cubic).

        Returns
        -------
        callable
            Spline interpolator function.
        """
        if self.extrapolate:
            bc_type = "natural"
        else:
            bc_type = None
        return make_interp_spline(x, y, k=k, bc_type=bc_type)

    def _get_interpolator(self, method: str) -> Callable:
        """
        Get the appropriate interpolator for the specified method.

        Parameters
        ----------
        method : str
            Interpolation method name.

        Returns
        -------
        callable
            Interpolator function.
        """
        x, y = self.x, self.y
        if method == "pchip":
            return self._get_pchip_interpolator(x, y)
        elif method in ["cubic spline", "linear"]:
            k = 3 if method == "cubic spline" else 1
            # Check if we have enough points for the spline degree
            if len(x) <= k:
                raise ValueError(
                    f"Need at least {k + 1} points for {method} interpolation, got {len(x)}"
                )
            return self._get_spline_interpolator(x, y, k=k)
        else:
            raise ValueError(f"Invalid method: {method}")

    def _check_method(self, method: str | None):
        """
        Validate the interpolation method.

        Parameters
        ----------
        method : str, optional
            Method to validate.
        """
        if method is None:
            return

        default_methods = self._get_default_methods()
        if method not in default_methods:
            raise ValueError(
                f"Invalid method: {method}. Must be one of {', '.join(default_methods)}"
            )

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
        return f"Interpolator1D(method={self.method})"


def interp1d(
    xi: np.ndarray | Number,
    x: np.ndarray,
    y: np.ndarray,
    method: str | None = None,
    fill_value: Number | str | None = None,
    force_monotonic: bool | None = None,
) -> np.ndarray:
    """
    Convenience function for 1D interpolation.

    Parameters
    ----------
    xi : np.ndarray or number
        x-coordinates at which to interpolate. Can be scalar or array.
    x : np.ndarray
        x-coordinates of the data points. Must be 1D array.
    y : np.ndarray
        y-coordinates of the data points. Must have same length as x.
    method : str, optional
        Interpolation method. Options: "pchip" (default), "linear", "cubic spline".
    fill_value : number or str, optional
        Value to use for out-of-bounds interpolation.
        Can be a number or "extrapolate". Default is np.nan.
    force_monotonic : bool, optional
        Whether to enforce monotonicity by removing non-increasing points.
        Default is False.

    Returns
    -------
    np.ndarray
        Interpolated values at xi.
    """
    return Interpolator1D(x, y, method, fill_value, force_monotonic)(xi)
