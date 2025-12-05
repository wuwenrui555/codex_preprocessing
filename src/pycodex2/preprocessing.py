# %%
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pycodex2.constants import TQDM_FORMAT


def combine_csv_data(
    data_fs: list[Union[Path, str]],
    data_ids: list[str],
    marker_names: list[str],
    rename_dict: dict = None,
    col_cell_id: str = "cell_id",
    col_data_id: str = "data_id",
) -> ad.AnnData:
    """
    Combine multiple CSV data files into a single AnnData object.

    Parameters
    ----------
    data_fs : list[Union[Path, str]]
        List of file paths to the CSV data files.
    data_ids : list[str]
        List of identifiers corresponding to each data file.
    marker_names : list[str]
        List of marker names to be used as features.
    rename_dict : dict, optional
        Dictionary for renaming columns in the data files.
    col_cell_id : str, optional
        Column name for cell identifiers.
    col_data_id : str, optional
        Column name for data identifiers.

    Returns
    -------
    ad.AnnData
        Combined AnnData object.
    """
    adata_list = []

    for data_f, data_id in tqdm(
        zip(data_fs, data_ids),
        desc="Reading data",
        bar_format=TQDM_FORMAT,
        total=len(data_fs),
    ):
        data = pd.read_csv(data_f)

        if rename_dict is not None:
            data = data.rename(columns=rename_dict)

        data["index"] = [f"{data_id}_c{i}" for i in data[col_cell_id]]
        data[col_data_id] = data_id

        data_X = data[["index"] + marker_names].set_index("index")
        data_meta = data.drop(columns=marker_names).set_index("index")

        adata = ad.AnnData(X=data_X, obs=data_meta)
        adata_list.append(adata)

    adata_combined = ad.concat(adata_list)

    return adata_combined


class ExtremeCutoff:
    """
    A class for detecting and filtering extreme values using statistical methods.

    This class calculates cutoff thresholds based on mean/median and standard
    deviation/MAD (Median Absolute Deviation) in log10 space, useful for
    identifying outliers in single-cell data quality control (PMID: 32873325).

    Parameters
    ----------
    values : Union[np.ndarray, list]
        Array or list of numerical values to analyze.

    Attributes
    ----------
    values : np.ndarray
        Original input values as numpy array.
    n_total : int
        Total number of values.
    log : np.ndarray
        Log10-transformed values (log10(values + 1)).
    log_mean : float
        Mean of log-transformed values.
    log_std : float
        Standard deviation of log-transformed values.
    log_median : float
        Median of log-transformed values.
    log_mad : float
        Median Absolute Deviation of log-transformed values.

    Examples
    --------
    >>> values = adata.obs["total_counts"].values
    >>> cutoff = ExtremeCutoff(values)
    >>> print(cutoff)
    >>> cutoff.test_cutoffs(n_sigmas=[1, 2, 3])
    >>> cutoff.plot_cutoffs(n_sigmas=[2, 3, 5])
    >>> mask = cutoff.filter_values(method="median", n_sigma=3)
    >>> adata_filtered = adata[mask]
    """

    def __init__(self, values: Union[np.ndarray, list]):
        self.values = np.array(values)

        # Validation
        if len(self.values) == 0:
            raise ValueError("Input values cannot be empty.")

        if np.any(np.isnan(self.values)):
            raise ValueError(
                "Input values contain NaN. Please remove or impute NaN values first."
            )

        if np.any(np.isinf(self.values)):
            raise ValueError("Input values contain infinite values.")

        if np.any(self.values < 0):
            raise ValueError(
                "Input values contain negative values. "
                "log10 transformation requires non-negative values."
            )

        # Calculate statistics
        self.n_total = len(self.values)
        self.log = np.log10(self.values + 1)
        self.log_mean = np.mean(self.log)
        self.log_std = np.std(self.log)
        self.log_median = np.median(self.log)
        self.log_mad = np.median(np.abs(self.log - self.log_median))

        # Warn if MAD is zero
        if self.log_mad == 0:
            warnings.warn(
                "MAD is zero, indicating all values are identical or very similar. "
                "Filtering may not work as expected.",
                UserWarning,
            )

    def __repr__(self):
        return (
            f"ExtremeCutoff(n={self.n_total}, log_mean={self.log_mean:.2f}, "
            f"log_std={self.log_std:.2f}, log_median={self.log_median:.2f}, "
            f"log_mad={self.log_mad:.2f})"
        )

    def _calculate_statistics(
        self,
        method: str = "median",
        n_sigma: Union[int, float, None] = 3,
        n_sigma_lower: Union[int, float, None] = None,
        n_sigma_upper: Union[int, float, None] = None,
    ) -> Tuple[float, float, float, float, int, int, float, float]:
        """
        Calculate cutoff thresholds and filtering statistics.

        Parameters
        ----------
        method : str, optional
            Statistical method to use. Must be 'mean' or 'median'. Default is 'median'.
        n_sigma : int or float, optional
            Number of standard deviations/MADs for symmetric cutoffs. Default is 3.
        n_sigma_lower : int or float, optional
            Number of standard deviations/MADs for lower cutoff. If None, uses n_sigma.
        n_sigma_upper : int or float, optional
            Number of standard deviations/MADs for upper cutoff. If None, uses n_sigma.

        Returns
        -------
        tuple
            A tuple containing:
            - lower_log (float): Lower threshold in log10 space
            - upper_log (float): Upper threshold in log10 space
            - lower (float): Lower threshold in original space
            - upper (float): Upper threshold in original space
            - n_filtered_lower (int): Number of values below lower threshold
            - n_filtered_upper (int): Number of values above upper threshold
            - perc_filtered_lower (float): Percentage of values below lower threshold
            - perc_filtered_upper (float): Percentage of values above upper threshold

        Raises
        ------
        ValueError
            If method is not 'mean' or 'median', or if n_sigma parameters are invalid.
        """
        if method not in ["mean", "median"]:
            raise ValueError("Method must be 'mean' or 'median'.")

        if (n_sigma_lower is None or n_sigma_upper is None) and n_sigma is None:
            raise ValueError(
                "n_sigma must be provided if n_sigma_lower or n_sigma_upper is None."
            )

        if n_sigma_lower is None:
            n_sigma_lower = n_sigma
        if n_sigma_upper is None:
            n_sigma_upper = n_sigma

        if method == "mean":
            lower_log = self.log_mean - n_sigma_lower * self.log_std
            upper_log = self.log_mean + n_sigma_upper * self.log_std
        elif method == "median":
            lower_log = self.log_median - n_sigma_lower * self.log_mad
            upper_log = self.log_median + n_sigma_upper * self.log_mad

        lower = 10**lower_log - 1
        upper = 10**upper_log - 1

        n_filtered_lower = np.sum(self.values < lower)
        n_filtered_upper = np.sum(self.values > upper)
        perc_filtered_lower = n_filtered_lower / self.n_total * 100
        perc_filtered_upper = n_filtered_upper / self.n_total * 100

        return (
            lower_log,
            upper_log,
            lower,
            upper,
            n_filtered_lower,
            n_filtered_upper,
            perc_filtered_lower,
            perc_filtered_upper,
        )

    def get_thresholds(
        self,
        method: str = "median",
        n_sigma: Union[int, float] = 3,
        n_sigma_lower: Union[int, float, None] = None,
        n_sigma_upper: Union[int, float, None] = None,
    ) -> dict:
        """
        Get cutoff thresholds and filtering statistics.

        Parameters
        ----------
        method : str, optional
            Statistical method to use. Default is 'median'.
        n_sigma : int or float, optional
            Number of standard deviations/MADs for symmetric cutoffs. Default is 3.
        n_sigma_lower : int or float, optional
            Number of standard deviations/MADs for lower cutoff.
        n_sigma_upper : int or float, optional
            Number of standard deviations/MADs for upper cutoff.

        Returns
        -------
        dict
            Dictionary containing threshold values and filtering statistics.

        Examples
        --------
        >>> cutoff = ExtremeCutoff(values)
        >>> thresholds = cutoff.get_thresholds(method="median", n_sigma=3)
        >>> print(f"Lower: {thresholds['lower']:.2f}, Upper: {thresholds['upper']:.2f}")
        """
        (
            lower_log,
            upper_log,
            lower,
            upper,
            n_filtered_lower,
            n_filtered_upper,
            perc_filtered_lower,
            perc_filtered_upper,
        ) = self._calculate_statistics(
            method=method,
            n_sigma=n_sigma,
            n_sigma_lower=n_sigma_lower,
            n_sigma_upper=n_sigma_upper,
        )

        return {
            "lower": lower,
            "upper": upper,
            "lower_log": lower_log,
            "upper_log": upper_log,
            "n_filtered_lower": n_filtered_lower,
            "n_filtered_upper": n_filtered_upper,
            "perc_filtered_lower": perc_filtered_lower,
            "perc_filtered_upper": perc_filtered_upper,
        }

    def test_cutoffs(
        self,
        n_sigmas: List[Union[int, float]] = None,
        return_df: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """
        Test and display cutoff thresholds for different sigma values.

        Prints a formatted table showing lower and upper cutoff values along with
        the percentage of data points that would be filtered at each threshold.

        Parameters
        ----------
        n_sigmas : list[int or float], optional
            List of sigma values to test. Default is [1, 2, 3].
        return_df : bool, optional
            If True, return the results DataFrame. Default is False.

        Returns
        -------
        None or pd.DataFrame
            If return_df is True, returns DataFrame. Otherwise returns None.

        Examples
        --------
        >>> cutoff = ExtremeCutoff(values)
        >>> cutoff.test_cutoffs(n_sigmas=[1, 2, 3, 5])
        >>> df = cutoff.test_cutoffs(n_sigmas=[2, 3], return_df=True)
        """
        if n_sigmas is None:
            n_sigmas = [1, 2, 3]

        results = []
        for n_sigma in n_sigmas:
            for method in ["mean", "median"]:
                (
                    _,
                    _,
                    lower,
                    upper,
                    _,
                    _,
                    perc_filtered_lower,
                    perc_filtered_upper,
                ) = self._calculate_statistics(method=method, n_sigma=n_sigma)
                results.append(
                    (
                        f"{n_sigma}",
                        f"{method}",
                        f"{lower:.2f} ({perc_filtered_lower:.3g}%)",
                        f"{upper:.2f} ({perc_filtered_upper:.3g}%)",
                    )
                )

        results_df = pd.DataFrame(
            results,
            columns=["N Sigma", "Method", "Lower (Filtered%)", "Upper (Filtered%)"],
        )

        if return_df:
            return results_df

        print(
            results_df.to_markdown(
                index=False,
                tablefmt="psql",
                colalign=("center", "center", "right", "right"),
            )
        )

    def plot_cutoffs(
        self,
        n_sigmas: List[Union[int, float]] = None,
        bins: int = 100,
        kde: bool = False,
        linewidth: Union[int, float] = 2,
        figsize: Tuple[int, int] = (16, 6),
        show: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize distribution and cutoff thresholds.

        Creates two side-by-side histograms showing the distribution of log-transformed
        values with cutoff thresholds for mean-based and median-based methods.

        Parameters
        ----------
        n_sigmas : list[int or float], optional
            List of sigma values to plot. Default is [1, 2, 3].
        bins : int, optional
            Number of histogram bins. Default is 100.
        kde : bool, optional
            Whether to overlay kernel density estimation. Default is False.
        linewidth : int or float, optional
            Width of cutoff threshold lines. Default is 2.
        figsize : tuple[int, int], optional
            Figure size as (width, height). Default is (16, 6).
        show : bool, optional
            If True, display the plot. Default is True.
        **kwargs
            Additional keyword arguments passed to sns.histplot.

        Returns
        -------
        tuple
            (fig, axes) matplotlib figure and axes objects.

        Examples
        --------
        >>> cutoff = ExtremeCutoff(values)
        >>> fig, axes = cutoff.plot_cutoffs(n_sigmas=[2, 3], bins=50, kde=True)
        >>> fig.savefig('cutoffs.png', dpi=300, bbox_inches='tight')
        """
        if n_sigmas is None:
            n_sigmas = [1, 2, 3]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        colors = plt.cm.tab10(range(len(n_sigmas)))

        for i, method in enumerate(["mean", "median"]):
            ax = axes[i]
            sns.histplot(self.log, bins=bins, kde=kde, color="white", **kwargs, ax=ax)
            for j, n_sigma in enumerate(n_sigmas):
                (
                    lower_log,
                    upper_log,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._calculate_statistics(method=method, n_sigma=n_sigma)
                ax.axvline(
                    lower_log,
                    color=colors[j],
                    linestyle="--",
                    label=f"±{n_sigma}",
                    linewidth=linewidth,
                )
                ax.axvline(
                    upper_log,
                    color=colors[j],
                    linestyle="--",
                    linewidth=linewidth,
                )
            ax.set_xlabel("Value (log10 scale)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Method: {method.capitalize()}")
            ax.legend(title="N Sigma")

        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axes

    def filter_values(
        self,
        method: str = "median",
        n_sigma: Union[int, float] = 3,
        n_sigma_lower: Union[int, float, None] = None,
        n_sigma_upper: Union[int, float, None] = None,
    ) -> np.ndarray:
        """
        Filter values based on calculated cutoff thresholds.

        Returns a boolean mask indicating which values fall within the acceptable range.

        Parameters
        ----------
        method : str, optional
            Statistical method to use. Must be 'mean' or 'median'. Default is 'median'.
        n_sigma : int or float, optional
            Number of standard deviations/MADs for symmetric cutoffs. Default is 3.
        n_sigma_lower : int or float, optional
            Number of standard deviations/MADs for lower cutoff. If None, uses n_sigma.
            This allows asymmetric filtering (e.g., 5 MADs below, 2.5 MADs above).
        n_sigma_upper : int or float, optional
            Number of standard deviations/MADs for upper cutoff. If None, uses n_sigma.

        Returns
        -------
        np.ndarray
            Boolean array of the same length as input values, where True indicates
            values within the acceptable range (to keep), and False indicates outliers
            (to filter out).

        Examples
        --------
        >>> cutoff = ExtremeCutoff(adata.obs["total_counts"].values)
        >>> # Symmetric filtering: 3 MADs on both sides
        >>> mask = cutoff.filter_values(method="median", n_sigma=3)
        >>> adata_filtered = adata[mask]
        >>>
        >>> # Asymmetric filtering: more lenient on lower end (as in the paper)
        >>> mask = cutoff.filter_values(
        ...     method="median",
        ...     n_sigma_lower=5,
        ...     n_sigma_upper=2.5
        ... )
        >>> adata_filtered = adata[mask]
        """
        (
            _,
            _,
            lower,
            upper,
            _,
            _,
            _,
            _,
        ) = self._calculate_statistics(
            method=method,
            n_sigma=n_sigma,
            n_sigma_lower=n_sigma_lower,
            n_sigma_upper=n_sigma_upper,
        )
        index_keep = (self.values >= lower) & (self.values <= upper)
        return index_keep
