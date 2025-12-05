# %%
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pycodex2.constants import TQDM_FORMAT


################################################################################
# Combine CSV Data Files into AnnData Object
################################################################################
def combine_csv_data(
    data_fs: List[Union[Path, str]],
    data_ids: List[str],
    marker_names: List[str],
    rename_dict: Optional[Dict[str, str]] = None,
    col_cell_id: str = "cell_id",
    col_data_id: str = "data_id",
    validate_markers: bool = True,
) -> ad.AnnData:
    """
    Combine multiple CSV data files into a single AnnData object.

    This function reads multiple CSV files, standardizes their format, and combines
    them into a single AnnData object suitable for single-cell analysis. Each cell
    is assigned a unique identifier based on its data source and original cell ID.

    Parameters
    ----------
    data_fs : List[Union[Path, str]]
        List of file paths to the CSV data files.
    data_ids : List[str]
        List of identifiers corresponding to each data file. Must be unique and
        have the same length as data_fs.
    marker_names : List[str]
        List of marker names to be used as features (columns in the expression matrix).
    rename_dict : Dict[str, str], optional
        Dictionary for renaming columns in the data files before processing.
        Example: {'old_name': 'new_name'}. Default is None.
    col_cell_id : str, optional
        Column name for cell identifiers in the input CSV files. Default is "cell_id".
    col_data_id : str, optional
        Column name for data identifiers in the output AnnData.obs. Default is "data_id".
    validate_markers : bool, optional
        Whether to validate that all marker_names exist in each file. Default is True.

    Returns
    -------
    ad.AnnData
        Combined AnnData object with:
        - X: Expression matrix with marker_names as columns
        - obs: Metadata with unique cell indices and data_id column
        - var: Feature names (marker_names)

    Raises
    ------
    ValueError
        If input lists have mismatched lengths, contain duplicates, or if required
        columns are missing from CSV files.
    FileNotFoundError
        If any of the specified CSV files do not exist.

    Examples
    --------
    >>> # Basic usage
    >>> adata = combine_csv_data(
    ...     data_fs=['sample1.csv', 'sample2.csv'],
    ...     data_ids=['S1', 'S2'],
    ...     marker_names=['CD3', 'CD4', 'CD8']
    ... )
    >>> print(adata)
    >>> print(adata.obs['data_id'].value_counts())

    >>> # With column renaming
    >>> adata = combine_csv_data(
    ...     data_fs=['data1.csv', 'data2.csv'],
    ...     data_ids=['batch1', 'batch2'],
    ...     marker_names=['marker1', 'marker2'],
    ...     rename_dict={'old_marker1': 'marker1', 'old_marker2': 'marker2'}
    ... )

    Notes
    -----
    - Cell indices in the output are formatted as '{data_id}_c{cell_id}'
    - All columns except marker_names are stored in adata.obs
    - If a marker is missing after renaming, a ValueError is raised
    """
    # Input validation
    _validate_inputs(data_fs, data_ids, marker_names, col_cell_id, col_data_id)

    adata_list = []

    # Set up iterator
    iterator = tqdm(
        zip(data_fs, data_ids),
        desc="Reading data",
        bar_format=TQDM_FORMAT,
        total=len(data_fs),
    )

    # Process each file
    for data_f, data_id in iterator:
        try:
            adata = _process_single_file(
                data_f=data_f,
                data_id=data_id,
                marker_names=marker_names,
                rename_dict=rename_dict,
                col_cell_id=col_cell_id,
                col_data_id=col_data_id,
                validate_markers=validate_markers,
            )
            adata_list.append(adata)
        except Exception as e:
            raise RuntimeError(
                f"Error processing file '{data_f}' (data_id: '{data_id}'): {str(e)}"
            ) from e

    # Combine all AnnData objects
    try:
        adata_combined = ad.concat(adata_list, join="outer", merge="same")
    except Exception as e:
        raise RuntimeError(f"Error combining AnnData objects: {str(e)}") from e

    return adata_combined


def _validate_inputs(
    data_fs: List[Union[Path, str]],
    data_ids: List[str],
    marker_names: List[str],
    col_cell_id: str,
    col_data_id: str,
) -> None:
    """Validate input parameters."""
    # Check empty inputs
    if not data_fs:
        raise ValueError("data_fs cannot be empty.")
    if not data_ids:
        raise ValueError("data_ids cannot be empty.")
    if not marker_names:
        raise ValueError("marker_names cannot be empty.")

    # Check length match
    if len(data_fs) != len(data_ids):
        raise ValueError(
            f"Length of data_fs ({len(data_fs)}) must match "
            f"length of data_ids ({len(data_ids)})."
        )

    # Check for duplicate data_ids
    if len(data_ids) != len(set(data_ids)):
        duplicates = [id for id in data_ids if data_ids.count(id) > 1]
        raise ValueError(f"data_ids contains duplicates: {set(duplicates)}")

    # Check file existence
    for data_f in data_fs:
        path = Path(data_f)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {data_f}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {data_f}")

    # Check for invalid column names
    if not col_cell_id:
        raise ValueError("col_cell_id cannot be empty.")
    if not col_data_id:
        raise ValueError("col_data_id cannot be empty.")

    # Check marker_names validity
    if len(marker_names) != len(set(marker_names)):
        duplicates = [m for m in marker_names if marker_names.count(m) > 1]
        raise ValueError(f"marker_names contains duplicates: {set(duplicates)}")


def _process_single_file(
    data_f: Union[Path, str],
    data_id: str,
    marker_names: List[str],
    rename_dict: Optional[Dict[str, str]],
    col_cell_id: str,
    col_data_id: str,
    validate_markers: bool,
) -> ad.AnnData:
    """Process a single CSV file and return an AnnData object."""
    # Read CSV
    try:
        data = pd.read_csv(data_f)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}") from e

    if data.empty:
        raise ValueError(f"CSV file is empty: {data_f}")

    # Rename columns if needed
    if rename_dict is not None:
        data = data.rename(columns=rename_dict)

    # Validate required columns
    if col_cell_id not in data.columns:
        raise ValueError(
            f"Column '{col_cell_id}' not found in {data_f}. "
            f"Available columns: {list(data.columns)}"
        )

    # Check for duplicate cell IDs
    if data[col_cell_id].duplicated().any():
        n_duplicates = data[col_cell_id].duplicated().sum()
        warnings.warn(
            f"Found {n_duplicates} duplicate cell IDs in {data_f} (data_id: {data_id}). "
            f"This may cause unexpected behavior.",
            UserWarning,
        )

    # Validate markers
    if validate_markers:
        missing_markers = set(marker_names) - set(data.columns)
        if missing_markers:
            raise ValueError(
                f"Missing markers {missing_markers} in {data_f}. "
                f"Available columns: {list(data.columns)}"
            )
    else:
        # Only warn about missing markers
        missing_markers = set(marker_names) - set(data.columns)
        if missing_markers:
            warnings.warn(
                f"Missing markers {missing_markers} in {data_f}. "
                f"These will be filled with NaN.",
                UserWarning,
            )

    # Check for potential column conflicts
    if "index" in data.columns:
        warnings.warn(
            f"Column 'index' already exists in {data_f} and will be overwritten.",
            UserWarning,
        )

    # Create unique cell index
    data["index"] = [f"{data_id}_c{cell_id}" for cell_id in data[col_cell_id]]

    # Check for duplicate indices (shouldn't happen if cell_ids are unique per file)
    if data["index"].duplicated().any():
        raise ValueError(
            f"Duplicate indices created for {data_f}. "
            f"This usually indicates duplicate cell IDs in the file."
        )

    # Add data_id column
    if col_data_id in data.columns and col_data_id != "index":
        warnings.warn(
            f"Column '{col_data_id}' already exists in {data_f} and will be overwritten.",
            UserWarning,
        )
    data[col_data_id] = data_id

    # Split into expression matrix and metadata
    # Handle missing markers by using reindex
    available_markers = [m for m in marker_names if m in data.columns]
    data_X = data[["index"] + available_markers].set_index("index")

    # Reindex to include all markers (missing ones will be NaN)
    data_X = data_X.reindex(columns=marker_names)

    # Create metadata (exclude marker columns)
    meta_columns = [col for col in data.columns if col not in marker_names]
    data_meta = data[meta_columns].set_index("index")

    # Create AnnData object
    adata = ad.AnnData(
        X=data_X.values, obs=data_meta, var=pd.DataFrame(index=marker_names)
    )

    return adata


################################################################################
# Extreme Value Cutoff Detection and Filtering
################################################################################
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

        result_df = pd.DataFrame(
            {
                "Threshold": ["Lower", "Upper", "Total"],
                "Cutoff": [f"{lower:.2f}", f"{upper:.2f}", ""],
                "Filtered": [
                    f"{n_filtered_lower:,}",
                    f"{n_filtered_upper:,}",
                    f"{n_filtered_lower + n_filtered_upper:,}",
                ],
                "Filtered(%)": [
                    f"{perc_filtered_lower:.3g}%",
                    f"{perc_filtered_upper:.3g}%",
                    f"{perc_filtered_lower + perc_filtered_upper:.3g}%",
                ],
            }
        )
        print(
            result_df.to_markdown(
                index=False,
                tablefmt="psql",
                colalign=("center", "right", "right", "right"),
            )
        )

        index_keep = (self.values >= lower) & (self.values <= upper)
        return index_keep
