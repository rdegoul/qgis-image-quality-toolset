# -------------------------------------------------------------------------
# Copyright (C) 2025 Telespazio
# -------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import binned_statistic
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


# -------------------------------------------------------------------------
# Variogram implementation (replaces skgstat dependency)
# -------------------------------------------------------------------------

def _dowd_estimator(diffs):
    """
    Dowd (1984) robust semi-variance estimator.
    """
    return 2.198 * np.nanmedian(diffs) ** 2 / 2


def _spherical_model(h, r, c0, b=0.0):
    """
    Spherical variogram model.
    """
    h = np.asarray(h, dtype=float)
    return np.where(
        h <= r,
        b + c0 * (1.5 * (h / r) - 0.5 * (h / r) ** 3),
        b + c0,
    )


class _ModelWrapper:
    """Minimal wrapper to expose model.__name__ attribute."""

    def __init__(self, name):
        self.__name__ = name


class Variogram:
    """
    Lightweight variogram analysis class compatible with skgstat.Variogram.
    """

    def __init__(
        self,
        coordinates,
        values,
        estimator="dowd",
        model="spherical",
        n_lags=10,
        maxlag="median",
        use_nugget=True,
        fit_method="trf",
    ):
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.use_nugget = use_nugget
        self.fit_method = fit_method
        self._model_name = model
        self.model = _ModelWrapper(model)
        self._estimator_name = estimator
        self._n_lags = n_lags

        if estimator != "dowd":
            raise ValueError(f"Only 'dowd' estimator is supported, got '{estimator}'")
        if model != "spherical":
            raise ValueError(f"Only 'spherical' model is supported, got '{model}'")

        self._dist_condensed = pdist(self.coordinates, metric="euclidean")
        self._dist_matrix = None

        if maxlag == "median":
            self._maxlag = np.median(self._dist_condensed)
        elif maxlag is None:
            self._maxlag = np.max(self._dist_condensed)
        else:
            self._maxlag = float(maxlag)

        self.bins = np.linspace(0, self._maxlag, n_lags + 1)[1:]
        self.experimental = self._compute_experimental()

        self._parameters = None
        self._fit_success = False
        try:
            self._parameters = self._fit()
            self._fit_success = True
        except (ValueError, RuntimeError):
            if self.use_nugget:
                self._parameters = (np.mean(self.bins), np.mean(self.experimental), 0.0)
            else:
                self._parameters = (np.mean(self.bins), np.mean(self.experimental))

    @property
    def distance(self):
        return self._dist_condensed

    @property
    def pairwise_diffs(self):
        n = len(self.values)
        if self._dist_matrix is None:
            self._dist_matrix = squareform(self._dist_condensed)
        rows, cols = np.triu_indices(n, k=1)
        return np.abs(self.values[rows] - self.values[cols])

    @property
    def n_lags(self):
        return self._n_lags

    @property
    def maxlag(self):
        return self._maxlag

    @property
    def parameters(self):
        return self._parameters

    @property
    def fit_success(self):
        return self._fit_success

    def _compute_experimental(self):
        n = len(self.values)
        experimental = np.empty(len(self.bins))
        triu_rows, triu_cols = np.triu_indices(n, k=1)
        distances = self._dist_condensed
        diffs_all = np.abs(self.values[triu_rows] - self.values[triu_cols])

        for k, upper in enumerate(self.bins):
            lower = 0.0 if k == 0 else self.bins[k - 1]
            in_bin = (distances > lower) & (distances <= upper)

            if not np.any(in_bin):
                experimental[k] = np.nan
            else:
                bin_diffs = diffs_all[in_bin]
                experimental[k] = _dowd_estimator(bin_diffs)

        return experimental

    def _fit(self):
        valid = ~np.isnan(self.experimental)
        h = self.bins[valid]
        gamma = self.experimental[valid]

        if len(h) < 2:
            raise ValueError("Not enough valid lag classes to fit a model.")

        if self.use_nugget:
            def _model(h_val, r, c0, b):
                return _spherical_model(h_val, r, c0, b)

            p0 = [np.mean(h), np.max(gamma) - np.min(gamma), np.min(gamma)]
            bounds = (
                [0, 0, 0],
                [np.max(h) * 2, np.max(gamma) * 2, np.max(gamma)],
            )
        else:
            def _model(h_val, r, c0):
                return _spherical_model(h_val, r, c0, 0.0)

            p0 = [np.mean(h), np.max(gamma) - np.min(gamma)]
            bounds = (
                [0, 0],
                [np.max(h) * 2, np.max(gamma) * 2],
            )

        popt, _ = curve_fit(
            _model, h, gamma, p0=p0, bounds=bounds, method=self.fit_method
        )
        return tuple(popt)

    def fitted_model(self, h):
        if self.use_nugget:
            r, c0, b = self.parameters
        else:
            r, c0 = self.parameters
            b = 0.0
        return _spherical_model(h, r, c0, b)


def analyze_image_variogram(image, n_samples=5000, lag=None, plot=True):
    """
    Compute the image variogram and estimate SNR from the nugget effect.

    Args:
        image: 2D Array of radiance/reflectance values
        n_samples: Number of points to sample (recommended: 3000-5000)
        lag: Number of lag classes for the variogram
        plot: Whether to generate plots (not fully used in this function)

    Returns:
        A dictionary containing:
            V: Variogram object from skgstat
            coords: Sampled coordinates
            values: Sampled values
            snr: Estimated Signal-to-Noise Ratio
            sill: Variogram sill (total variance)
            nugget: Variogram nugget (noise variance)
            range_val: Variogram range (distance of spatial correlation)
            height: Image height
            width: Image width
    """
    
    height, width = image.shape
    print(f"Image: {height}×{width} = {height*width:,} pixels")
    
    # =========================================================================
    # STRATIFIED SAMPLING
    # =========================================================================
    # Stratified sampling ensures points are distributed across the entire ROI.
    
    # Divide the image into strata (grid of sub-regions)
    n_strata = int(np.sqrt(n_samples / 10))  # Ex: 22 strata → 484 sub-regions
    
    # Ensure n_strata is appropriate for image size
    n_strata = min(n_strata, min(height, width) // 2)
    n_strata = max(1, n_strata)
    
    strata_height = height // n_strata + 1
    strata_width = width // n_strata + 1
    points_per_strata = max(1, n_samples // (n_strata * n_strata))
    
    coords_list = []
    values_list = []
    
    for i in range(n_strata):
        for j in range(n_strata):
            # Strata boundaries
            y_start = i * strata_height
            y_end = min((i + 1) * strata_height, height)
            x_start = j * strata_width
            x_end = min((j + 1) * strata_width, width)

            # Skip invalid strata
            if y_end <= y_start or x_end <= x_start:
                continue
            
            # Sample randomly within this stratum
            n_points_strata = min(points_per_strata, (y_end - y_start) * (x_end - x_start))
            
            # Ensure we have valid dimensions for randint
            if n_points_strata <= 0 or y_end <= y_start or x_end <= x_start:
                continue

            y_sample = np.random.randint(y_start, y_end, n_points_strata)
            x_sample = np.random.randint(x_start, x_end, n_points_strata)
            
            # Coordinates in pixels
            coords_strata = np.column_stack([
                x_sample,
                y_sample
            ])
            values_strata = image[y_sample, x_sample]
            
            coords_list.append(coords_strata)
            values_list.append(values_strata)
    
    coords = np.vstack(coords_list)
    values = np.concatenate(values_list)

    # Check if we have any valid samples
    if len(values) == 0:
        raise ValueError("No valid samples could be extracted from the image. The image may be too small or contain only invalid values.")

    # Filter NaN values
    mask = ~np.isnan(values)
    coords = coords[mask]
    values = values[mask]
    
    # Check if we have any valid values after filtering NaN
    if len(values) == 0:
        raise ValueError("All sampled values are NaN. The image may contain only invalid data.")

    
    print(f"Sampled: {len(values):,} points (stratified: {n_strata}×{n_strata} strata)")
    print(f"Ratio: {len(values)/(height*width)*100:.2f}%")
    
    # =========================================================================
    # VARIOGRAM COMPUTATION
    # =========================================================================
    
    print("\nComputing variogram...")
    # Currently fixed to spherical model and dowd estimator as they are robust for SNR
    models = ['spherical']
    estimators = ['dowd']
    snr = None
    V = None
    range_val = None
    sill = None
    nugget = None

    for model in models:
        for estimator in estimators:
            print(f" - Model: {model}, Estimator: {estimator}")
            try:
                V = Variogram(
                    coordinates=coords,
                    values=values,
                    estimator=estimator,
                    model=model,
                    n_lags=lag,              # Number of distance classes
                    maxlag='median',        # Max distance = median of all distances
                    use_nugget=True,
                    fit_method='trf'        # Robust optimization method
                )
            except Exception as e:
                print(f"   Error during variogram fitting: {e}")
                continue

            range_val = V.parameters[0]
            sill = V.parameters[1] + V.parameters[2]      # Total variance (nugget + partial sill)
            gamma_exp = V.experimental
            # gamma_exp[0] is the semi-variance at lag 0 (nugget effect)
            # SNR = mean / sqrt(nugget semi-variance)
            nugget_semi_variance = float(gamma_exp[0])
            mean = np.nanmean(image)
            snr = mean / np.sqrt(nugget_semi_variance)
            nugget = 2 * nugget_semi_variance  # Full nugget variance (noise variance)
            print('SNR Computed from variogram: ', snr)
            print('Parameters fitted:', V.parameters)
            print('Nugget Computed from variogram: ', nugget)
            break
        
        if V is not None:
            break

    if V is None:
        print("   No variogram model could be fitted.")
        return None

    result = {
        'V': V,
        'coords': coords,
        'values': values,
        'snr': snr,
        'sill': sill,
        'nugget': nugget,
        'range_val': range_val,
        'height': height,
        'width': width,
    }

    return result
