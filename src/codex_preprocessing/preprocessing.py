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

from codex_preprocessing.constants import TQDM_FORMAT


################################################################################
# Processing History Tracking Helpers
################################################################################
def _record_processing_step(
    adata: ad.AnnData,
    step_name: str,
    parameters: Dict[str, any],
) -> None:
    """
    Record a processing step to the AnnData object's processing history.

    This function adds a new entry to adata.uns['processing_history'] containing
    the step name, timestamp, and parameters used. This allows tracking of all
    transformations applied to the data.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to record the processing step in.
    step_name : str
        Name of the processing step (e.g., 'nucleus_signal_normalization').
    parameters : dict
        Dictionary of parameters used for this processing step.

    Returns
    -------
    None
        Modifies adata.uns['processing_history'] in place.

    Examples
    --------
    >>> _record_processing_step(
    ...     adata,
    ...     'arcsinh_transformation',
    ...     {'cofactor': 5.0}
    ... )
    >>> print(adata.uns['processing_history'])
    """
    # Initialize processing history if it doesn't exist
    if "processing_history" not in adata.uns:
        adata.uns["processing_history"] = []

    # Create step entry
    step_entry = {
        "step": step_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "parameters": parameters,
    }

    # Append to history
    adata.uns["processing_history"].append(step_entry)


def _check_processing_step(
    adata: ad.AnnData,
    step_name: str,
) -> bool:
    """
    Check if a processing step has already been applied to the AnnData object.

    This function searches adata.uns['processing_history'] to determine whether
    a specific processing step has been previously applied.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to check.
    step_name : str
        Name of the processing step to check for (e.g., 'quantile_normalization').

    Returns
    -------
    bool
        True if the step has been applied before, False otherwise.

    Examples
    --------
    >>> if _check_processing_step(adata, 'arcsinh_transformation'):
    ...     print("Already transformed!")
    >>> else:
    ...     arcsinh_transformation(adata, cofactor=5.0, inplace=True)
    """
    # If no processing history exists, step hasn't been applied
    if "processing_history" not in adata.uns:
        return False

    # Check if step_name exists in any history entry
    return any(step["step"] == step_name for step in adata.uns["processing_history"])


def _get_processing_parameters(
    adata: ad.AnnData,
    step_name: str,
) -> Optional[Dict[str, any]]:
    """
    Get the parameters used for a specific processing step.

    This function retrieves the parameters that were used when a processing step
    was previously applied. If the step was applied multiple times, returns the
    parameters from the first occurrence.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to query.
    step_name : str
        Name of the processing step to retrieve parameters for.

    Returns
    -------
    dict or None
        Dictionary of parameters if the step was found, None otherwise.

    Examples
    --------
    >>> params = _get_processing_parameters(adata, 'quantile_normalization')
    >>> if params:
    ...     print(f"Min quantile: {params['min_quantile']}")
    ...     print(f"Max quantile: {params['max_quantile']}")
    """
    # If no processing history exists, return None
    if "processing_history" not in adata.uns:
        return None

    # Search for the step and return its parameters
    for step in adata.uns["processing_history"]:
        if step["step"] == step_name:
            return step["parameters"]

    # Step not found
    return None


def print_processing_history(adata: ad.AnnData) -> None:
    """
    Print a formatted summary of all processing steps applied to the AnnData object.

    This function provides a human-readable display of the processing history,
    including step names, timestamps, and parameters used for each transformation.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to display processing history for.

    Returns
    -------
    None
        Prints formatted output to console.
    """
    if "processing_history" not in adata.uns:
        print("No processing history found.")
        return

    if len(adata.uns["processing_history"]) == 0:
        print("Processing history is empty.")
        return

    print("\nProcessing History:")
    print("=" * 80)

    for i, step in enumerate(adata.uns["processing_history"], 1):
        print(f"\n{i}. {step['step']}")
        print(f"   Timestamp: {step['timestamp']}")
        print("   Parameters:")
        for key, value in step["parameters"].items():
            print(f"      {key}: {value}")

    print("\n" + "=" * 80 + "\n")


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
        filter_lower: bool = True,
        filter_upper: bool = True,
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
        filter_lower : bool, optional
            If True, calculate lower cutoff. Default is True.
        filter_upper : bool, optional
            If True, calculate upper cutoff. Default is True.

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
            center = self.log_mean
            spread = self.log_std
        elif method == "median":
            center = self.log_median
            spread = self.log_mad

        if filter_lower:
            lower_log = center - n_sigma_lower * spread
            lower = 10**lower_log - 1
        else:
            lower = np.min(self.values)
            lower_log = np.log10(lower + 1)

        if filter_upper:
            upper_log = center + n_sigma_upper * spread
            upper = 10**upper_log - 1
        else:
            upper = np.max(self.values)
            upper_log = np.log10(upper + 1)

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
                        f"{lower + upper:.2f} ({perc_filtered_lower + perc_filtered_upper:.3g}%)",
                    )
                )

        results_df = pd.DataFrame(
            results,
            columns=[
                "N Sigma",
                "Method",
                "Lower (Filtered%)",
                "Upper (Filtered%)",
                "Total (Filtered%)",
            ],
        )

        if return_df:
            return results_df

        print(
            results_df.to_markdown(
                index=False,
                tablefmt="psql",
                colalign=("center", "center", "right", "right", "right"),
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
        filter_lower: bool = True,
        filter_upper: bool = True,
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
        filter_lower : bool, optional
            If True, apply lower cutoff filtering. Default is True.
        filter_upper : bool, optional
            If True, apply upper cutoff filtering. Default is True.

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
            filter_lower=filter_lower,
            filter_upper=filter_upper,
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


################################################################################
# Nucleus Signal Normalization
################################################################################
def nucleus_signal_normalization(
    adata: ad.AnnData,
    col_data_id: str,
    marker_nucleus: str = "DAPI",
    method: str = "median",
    inplace: bool = False,
    skip_check: bool = False,
) -> Optional[ad.AnnData]:
    """
    Normalize single-cell expression data using nucleus marker signal intensity.

    This function performs batch-wise normalization by dividing all marker intensities
    by the central tendency (median or mean) of a nucleus marker (e.g., DAPI) within
    each batch. This approach corrects for technical variation in overall signal
    intensity across different batches or samples.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    col_data_id : str
        Column name in adata.obs that contains batch/sample identifiers for
        grouping cells during normalization.
    marker_nucleus : str, optional
        Name of the nucleus marker in adata.var_names to use for normalization.
        Default is "DAPI".
    method : str, optional
        Normalization method, either "median" or "mean". Determines which central
        tendency measure to use for computing normalization factors. Default is "median".
    inplace : bool, optional
        If True, modify the input AnnData object in place. If False, return a
        modified copy. Default is False.
    skip_check : bool, optional
        If True, skip the check for whether this normalization has already been
        applied. Use with caution as applying normalization multiple times may
        lead to incorrect results. Default is False.

    Returns
    -------
    ad.AnnData or None
        If inplace=False, returns a normalized copy of the input AnnData object.
        If inplace=True, returns None and modifies the input object directly.
        The output includes a new column 'nucleus_norm_factor' in adata.obs
        containing the normalization factor applied to each cell.

    Raises
    ------
    ValueError
        If method is not "median" or "mean", if marker_nucleus is not found in
        adata.var_names, or if multiple markers with the same name exist.
    RuntimeError
        If the normalization has already been applied and skip_check is False.

    Examples
    --------
    >>> # Basic usage with default median normalization
    >>> adata_norm = nucleus_signal_normalization(
    ...     adata=adata,
    ...     col_data_id='data_id',
    ...     marker_nucleus='DAPI'
    ... )
    >>> print(adata_norm.obs['nucleus_norm_factor'].describe())

    >>> # In-place normalization using mean
    >>> nucleus_signal_normalization(
    ...     adata=adata,
    ...     col_data_id='batch',
    ...     marker_nucleus='DAPI',
    ...     method='mean',
    ...     inplace=True
    ... )
    >>> print(adata.obs['nucleus_norm_factor'].head())

    Notes
    -----
    - Normalization is performed separately for each unique value in col_data_id
    - All marker intensities (adata.X) are divided by the normalization factor
    - The normalization factor for each batch is stored in adata.obs['nucleus_norm_factor']
    - Median normalization is generally more robust to outliers than mean normalization
    """
    if not inplace:
        adata = adata.copy()

    # Check if already processed
    if not skip_check:
        if _check_processing_step(adata, "nucleus_signal_normalization"):
            prev_params = _get_processing_parameters(
                adata, "nucleus_signal_normalization"
            )
            raise RuntimeError(
                f"nucleus_signal_normalization has already been applied with "
                f"parameters: {prev_params}. Set skip_check=True to bypass this check."
            )

    data_ids = adata.obs[col_data_id].unique()

    if method not in ["median", "mean"]:
        raise ValueError(
            f"Normalization method must be 'median' or 'mean', got {method}."
        )

    norm_fun = np.median if method == "median" else np.mean

    for data_id in tqdm(data_ids, desc="Nucleus Normalization", bar_format=TQDM_FORMAT):
        mask = adata.obs[col_data_id] == data_id
        marker_nucleus_idx = adata.var_names == marker_nucleus

        if sum(marker_nucleus_idx) == 0:
            raise ValueError(f"{marker_nucleus} not found in var_names")
        elif sum(marker_nucleus_idx) > 1:
            raise ValueError(f"Multiple {marker_nucleus} found in var_names")

        norm_factor = norm_fun(adata.X[mask, marker_nucleus_idx])
        adata.X[mask, :] /= norm_factor
        adata.obs.loc[mask, "nucleus_norm_factor"] = norm_factor

    # Record processing step
    _record_processing_step(
        adata,
        "nucleus_signal_normalization",
        {
            "col_data_id": col_data_id,
            "marker_nucleus": marker_nucleus,
            "method": method,
        },
    )

    if not inplace:
        return adata


################################################################################
# Arcsinh Transformation
################################################################################
def arcsinh_transformation(
    adata: ad.AnnData,
    cofactor: float = 1.0,
    inplace: bool = False,
    skip_check: bool = False,
) -> Optional[ad.AnnData]:
    """
    Apply arcsinh (inverse hyperbolic sine) transformation to expression data.

    This function performs arcsinh transformation on the expression matrix, which
    is commonly used in single-cell proteomics and flow cytometry data analysis
    to stabilize variance and make data more normally distributed. The transformation
    is defined as arcsinh(x / cofactor), where the cofactor controls the linearity
    of the transformation for small values.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    cofactor : float, optional
        Cofactor (scaling factor) for the arcsinh transformation. Larger values
        make the transformation more linear for small values, while smaller values
        make it more log-like. Typical values range from 0.001 to 1000, depending
        on the data scale. Default is 1.0.
    inplace : bool, optional
        If True, modify the input AnnData object in place. If False, return a
        transformed copy. Default is False.
    skip_check : bool, optional
        If True, skip the check for whether this transformation has already been
        applied. Use with caution as applying the same transformation multiple
        times may lead to incorrect results. Default is False.

    Returns
    -------
    ad.AnnData or None
        If inplace=False, returns a transformed copy of the input AnnData object.
        If inplace=True, returns None and modifies the input object directly.

    Raises
    ------
    RuntimeError
        If the transformation has already been applied and skip_check is False.

    Examples
    --------
    >>> # Basic usage with default cofactor
    >>> adata_transformed = arcsinh_transformation(adata, cofactor=1.0)
    >>> print(adata_transformed.X.min(), adata_transformed.X.max())

    >>> # In-place transformation with custom cofactor
    >>> arcsinh_transformation(adata, cofactor=5.0, inplace=True)
    >>> print(adata.X[:5, :3])

    >>> # Test different cofactors to find optimal value
    >>> plot_arcsinh_transformation(
    ...     adata,
    ...     marker_name='CD3',
    ...     cofactors=[0.1, 1, 10, 100]
    ... )
    >>> adata_transformed = arcsinh_transformation(adata, cofactor=5.0)

    Notes
    -----
    - The arcsinh transformation is defined as: arcsinh(x / cofactor)
    - Use plot_arcsinh_transformation() to visualize and select appropriate cofactor
    """
    if not inplace:
        adata = adata.copy()

    if not skip_check:
        if _check_processing_step(adata, "arcsinh_transformation"):
            prev_params = _get_processing_parameters(adata, "arcsinh_transformation")
            raise RuntimeError(
                f"arcsinh_transformation has already been applied with "
                f"parameters: {prev_params}. Set skip_check=True to bypass this check."
            )

    adata.X = np.arcsinh(adata.X / cofactor)

    # Record processing step
    _record_processing_step(
        adata,
        "arcsinh_transformation",
        {
            "cofactor": cofactor,
        },
    )

    if not inplace:
        return adata


def plot_arcsinh_transformation(
    adata: ad.AnnData,
    marker_name: str,
    cofactors: Union[List[float], int, float] = None,
    hue: Optional[str] = None,
    legend: bool = True,
    nrow: Optional[int] = None,
    figsize_sub: Tuple[int, int] = (6, 4),
    show: bool = True,
    sample_size: Optional[int] = None,
    seed: int = 81,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize the effect of arcsinh transformation with different cofactors.

    This function creates a grid of density plots showing how different cofactor
    values affect the distribution of a specific marker. This is useful for
    selecting an appropriate cofactor for the arcsinh transformation.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    marker_name : str
        Name of the marker in adata.var_names to visualize.
    cofactors : list[float] or int or float, optional
        List of cofactor values to test. Default is [1000, 100, 10, 1, 0.1,
        0.01, 0.001, 0.0001]. If a single int or float is provided, it will be
        converted to a list with one element.
    hue : str, optional
        Column name in adata.obs to use for color grouping in density plots.
        Useful for comparing distributions across batches or conditions.
        Default is None, meaning no grouping.
    legend : bool, optional
        Whether to display legend when hue is specified. Default is True.
    nrow : int, optional
        Number of rows in the subplot grid. If None, automatically calculated
        as the square root of the number of cofactors. Default is None.
    figsize_sub : tuple[int, int], optional
        Size of each individual subplot as (width, height). Default is (6, 4).
    show : bool, optional
        If True, display the plot. Default is True.
    sample_size : int, optional
        Number of cells to randomly sample for plotting. Useful for large datasets
        to speed up visualization. If None, use all cells. Default is None.
    seed : int, optional
        Random seed for reproducible sampling when sample_size is specified.
        Default is 81.
    **kwargs
        Additional keyword arguments passed to sns.kdeplot.

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes objects.

    Raises
    ------
    ValueError
        If marker_name is not found in adata.var_names, if multiple markers with
        the same name exist, or if hue column is not found in adata.obs.

    Examples
    --------
    >>> # Basic usage to compare different cofactors
    >>> fig, axes = plot_arcsinh_transformation(
    ...     adata,
    ...     marker_name='CD3',
    ...     cofactors=[1, 5, 10, 50]
    ... )

    >>> # Compare distributions across batches with custom layout
    >>> fig, axes = plot_arcsinh_transformation(
    ...     adata,
    ...     marker_name='CD4',
    ...     cofactors=[0.1, 1, 10, 100],
    ...     hue='data_id',
    ...     nrow=2,
    ...     sample_size=10000
    ... )
    >>> fig.savefig('cd4_cofactor_comparison.png', dpi=300, bbox_inches='tight')

    Notes
    -----
    - Smaller cofactors compress low values more (more log-like)
    - Larger cofactors preserve linearity for small values
    - Sampling is recommended for datasets with >100,000 cells to improve performance
    - The title shows the cofactor value and sampling information if applicable
    """
    if cofactors is None:
        cofactors = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]

    if isinstance(cofactors, (int, float)):
        cofactors = [cofactors]

    # Calculate grid dimensions
    n_cofactors = len(cofactors)
    if nrow is None:
        nrow = int(np.sqrt(n_cofactors))
    ncol = n_cofactors // nrow + (1 if n_cofactors % nrow != 0 else 0)

    fig, axes = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(figsize_sub[0] * ncol, figsize_sub[1] * nrow),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Hide unused subplots
    for i in range(n_cofactors, len(axes_flat)):
        axes_flat[i].axis("off")

    # Plot each cofactor
    for i, cofactor in enumerate(cofactors):
        ax = axes_flat[i]
        _ax_kdeplot_arcsinh_transformation(
            adata,
            marker_name,
            cofactor=cofactor,
            hue=hue,
            legend=legend,
            ax=ax,
            sample_size=sample_size,
            seed=seed,
            **kwargs,
        )

    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Return single axes if only one cofactor, array for multiple cofactors
    if n_cofactors == 1:
        return fig, axes_flat[0]
    else:
        return fig, axes_flat[:n_cofactors]


def _ax_kdeplot_arcsinh_transformation(
    adata: ad.AnnData,
    marker_name: str,
    cofactor: float = 1.0,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    sample_size: Optional[int] = None,
    seed: int = 81,
    **kwargs,
) -> None:
    """
    Helper function to plot arcsinh-transformed marker distribution on a single axis.


    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    marker_name : str
        Name of the marker in adata.var_names to visualize.
    cofactor : float, optional
        Cofactor for the arcsinh transformation. Default is 1.0.
    hue : str, optional
        Column name in adata.obs for color grouping. Default is None.
    title : str, optional
        Custom title for the plot. If None, automatically generated from
        marker_name and cofactor. Default is None.
    legend : bool, optional
        Whether to display legend. Default is True.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes. Default is None.
    sample_size : int, optional
        Number of cells to randomly sample. If None, use all cells. Default is None.
    seed : int, optional
        Random seed for sampling. Default is 81.
    **kwargs
        Additional keyword arguments passed to sns.kdeplot.

    Raises
    ------
    ValueError
        If marker_name is not found in adata.var_names, if multiple markers with
        the same name exist, or if hue column is not found in adata.obs.
    """
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = f"{marker_name}, cofactor={cofactor}"

    # Validate marker exists
    if sum(adata.var_names == marker_name) == 0:
        raise ValueError(f"{marker_name} not found in var_names")
    if sum(adata.var_names == marker_name) > 1:
        raise ValueError(f"Multiple {marker_name} found in var_names")

    # Validate hue column
    if hue is not None and hue not in adata.obs.columns:
        raise ValueError(f"{hue} not found in obs columns")

    # Downsample if requested
    if sample_size is not None and adata.n_obs > sample_size:
        title = (
            title + f", downsampled={sample_size:.1e} ({sample_size / adata.n_obs:.2f})"
        )
        adata = _downsample_cells(adata, sample_size=sample_size, seed=seed)

    # Create KDE plot
    sns.kdeplot(
        x=np.arcsinh(adata[:, marker_name].X.flatten() / cofactor),
        hue=adata.obs[hue] if hue is not None else None,
        legend=legend and (hue is not None),
        ax=ax,
        **kwargs,
    )
    if legend and (hue is not None):
        leg = ax.get_legend()
        leg.set_title(hue)
        leg.set_loc("best")
        for text in leg.get_texts():
            text.set_fontsize("small")

    ax.set_xlabel(f"arcsinh({marker_name} / {cofactor})")
    ax.set_title(title)


def _downsample_cells(
    adata: ad.AnnData,
    sample_size: int,
    replace: bool = False,
    seed: int = 81,
) -> ad.AnnData:
    """
    Randomly downsample cells from an AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    sample_size : int
        Number of cells to randomly sample.
    replace : bool, optional
        Whether to sample with replacement. Default is False.
    seed : int, optional
        Random seed for reproducible sampling. Default is 81.

    Returns
    -------
    ad.AnnData
        A new AnnData object containing the sampled subset of cells.
        If sample_size >= adata.n_obs, returns the original adata unchanged.

    Examples
    --------
    >>> adata_small = _downsample_cells(adata, sample_size=10000)
    >>> print(f"Original: {adata.n_obs}, Downsampled: {adata_small.n_obs}")
    """
    if sample_size >= adata.n_obs:
        return adata

    np.random.seed(seed)
    sampled_indices = np.random.choice(adata.n_obs, size=sample_size, replace=replace)
    return adata[sampled_indices]


################################################################################
# Quantile Normalization
################################################################################
def quantile_normalization(
    adata: ad.AnnData,
    min_quantile: float = 0.01,
    max_quantile: float = 0.99,
    equal_return: float = 0.0,
    inplace: bool = True,
    skip_check: bool = False,
) -> Optional[ad.AnnData]:
    """
    Normalize marker expression to [0, 1] range using quantile-based clipping.

    This function performs marker-wise normalization by mapping values between
    specified quantiles to the [0, 1] range. Values below the minimum quantile
    are set to 0, and values above the maximum quantile are set to 1. This approach
    is robust to outliers and useful for standardizing marker intensities across
    different scales.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    min_quantile : float, optional
        Lower quantile threshold for normalization. Values below this quantile
        are clipped to 0. Must be between 0 and 1. Default is 0.01 (1st percentile).
    max_quantile : float, optional
        Upper quantile threshold for normalization. Values above this quantile
        are clipped to 1. Must be between 0 and 1 and greater than min_quantile.
        Default is 0.99 (99th percentile).
    equal_return : float, optional
        Value to return when min and max quantiles are equal (i.e., all values
        are identical). Default is 0.0.
    inplace : bool, optional
        If True, modify the input AnnData object in place. If False, return a
        normalized copy. Default is True.
    skip_check : bool, optional
        If True, skip the check for whether this normalization has already been
        applied. Use with caution as applying normalization multiple times may
        lead to incorrect results. Default is False.

    Returns
    -------
    ad.AnnData or None
        If inplace=False, returns a normalized copy of the input AnnData object.
        If inplace=True, returns None and modifies the input object directly.
        All marker values are scaled to [0, 1] range.

    Raises
    ------
    ValueError
        If min_quantile >= max_quantile, or if quantile values are outside [0, 1].
    RuntimeError
        If the normalization has already been applied and skip_check is False.

    Examples
    --------
    >>> # Basic usage with default 1st and 99th percentiles
    >>> adata_norm = quantile_normalization(adata, inplace=False)

    >>> # In-place normalization with custom quantiles
    >>> quantile_normalization(
    ...     adata,
    ...     min_quantile=0.05,
    ...     max_quantile=0.95,
    ...     inplace=True
    ... )
    >>> print(adata[:, 'CD3'].X.min(), adata[:, 'CD3'].X.max())

    Notes
    -----
    - Normalization is performed independently for each marker
    - This method is robust to extreme outliers compared to min-max normalization
    - If all values for a marker are identical, they are set to equal_return
    - Values are linearly scaled between the quantile thresholds
    """
    if not inplace:
        adata = adata.copy()

    # Check if already processed
    if not skip_check:
        if _check_processing_step(adata, "quantile_normalization"):
            prev_params = _get_processing_parameters(adata, "quantile_normalization")
            raise RuntimeError(
                f"quantile_normalization has already been applied with "
                f"parameters: {prev_params}. Set skip_check=True to bypass this check."
            )

    # Validate quantile parameters
    if not (0 <= min_quantile <= 1):
        raise ValueError(f"min_quantile must be between 0 and 1, got {min_quantile}")

    if not (0 <= max_quantile <= 1):
        raise ValueError(f"max_quantile must be between 0 and 1, got {max_quantile}")

    if min_quantile >= max_quantile:
        raise ValueError(
            f"min_quantile ({min_quantile}) must be less than "
            f"max_quantile ({max_quantile})"
        )

    # Initialize normalized data matrix
    X_norm = np.zeros_like(adata.X, dtype=float)

    # Normalize each marker
    for i, marker in enumerate(
        tqdm(adata.var_names, desc="Quantile Normalization", bar_format=TQDM_FORMAT)
    ):
        x = adata.X[:, i].flatten()
        x_min = np.quantile(x, min_quantile)
        x_max = np.quantile(x, max_quantile)

        # Handle edge case where min and max are equal
        if x_min == x_max:
            warnings.warn(
                f"Marker '{marker}' has identical values at {min_quantile} and "
                f"{max_quantile} quantiles. Setting all values to {equal_return}.",
                UserWarning,
            )
            x_norm = np.full_like(x, equal_return, dtype=float)
        else:
            # Normalize values to [0, 1] range
            x_norm = (x - x_min) / (x_max - x_min)
            x_norm = np.clip(x_norm, 0, 1)

        X_norm[:, i] = x_norm

    adata.X = X_norm

    # Record processing step
    _record_processing_step(
        adata,
        "quantile_normalization",
        {
            "min_quantile": min_quantile,
            "max_quantile": max_quantile,
            "equal_return": equal_return,
        },
    )

    if not inplace:
        return adata


def plot_quantile_normalization(
    adata: ad.AnnData,
    marker_names: Union[str, List[str]],
    quantiles: List[float] = None,
    sample_size: Optional[int] = None,
    seed: int = 81,
    linewidth: Union[int, float] = 1,
    show: bool = True,
    legend: bool = True,
    nrow: Optional[int] = None,
    figsize_sub: Tuple[int, int] = (6, 4),
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Visualize quantile cutoff positions on marker distributions.

    This function creates density plots showing the distribution of one or more
    markers with vertical lines indicating the positions of specified quantiles.
    This is useful for selecting appropriate quantile thresholds for normalization.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing single-cell expression data.
    marker_names : str or list[str]
        Name(s) of marker(s) in adata.var_names to visualize. Can be a single
        marker name as a string or a list of marker names.
    quantiles : list[float], optional
        List of quantile values (between 0 and 1) to display as vertical lines.
        Default is [0.001, 0.005, 0.01, 0.05, 0.95, 0.99, 0.995, 0.999].
    sample_size : int, optional
        Number of cells to randomly sample for plotting. Useful for large datasets
        to speed up visualization. If None, use all cells. Default is None.
    seed : int, optional
        Random seed for reproducible sampling when sample_size is specified.
        Default is 81.
    linewidth : int or float, optional
        Width of the quantile threshold lines. Default is 1.
    show : bool, optional
        If True, display the plot. If False, close the plot and return the
        figure object. Default is True.
    legend : bool, optional
        Whether to display legend with quantile values. Default is True.
    nrow : int, optional
        Number of rows in the subplot grid when plotting multiple markers.
        If None, automatically calculated as the square root of the number
        of markers. Default is None.
    figsize_sub : tuple[int, int], optional
        Size of each individual subplot as (width, height). Default is (6, 4).

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes objects. When plotting a single
        marker, axes is a single Axes object. When plotting multiple markers,
        axes is a numpy array of Axes objects.

    Raises
    ------
    ValueError
        If any marker_name is not found in adata.var_names or if multiple markers
        with the same name exist.

    Examples
    --------
    >>> # Single marker visualization
    >>> fig, ax = plot_quantile_normalization(adata, marker_names='CD3')

    >>> # Multiple markers with custom quantiles
    >>> fig, axes = plot_quantile_normalization(
    ...     adata,
    ...     marker_names=['CD3', 'CD4', 'CD8', 'CD45'],
    ...     quantiles=[0.01, 0.05, 0.95, 0.99],
    ...     sample_size=50000
    ... )

    Notes
    -----
    - Quantile lines show both the quantile value and the actual data value
    - Sampling is recommended for datasets with >100,000 cells to improve performance
    - Use this plot to identify appropriate quantile thresholds before normalization
    - For single marker plots, axes is returned as a single object, not an array
    """
    if quantiles is None:
        quantiles = [0.001, 0.005, 0.01, 0.05, 0.95, 0.99, 0.995, 0.999]

    # Validate quantiles is a list
    if not isinstance(quantiles, list):
        raise ValueError(
            "quantiles must be a list of quantile values, "
            f"got {type(quantiles).__name__}"
        )

    # Convert single marker to list for uniform processing
    if isinstance(marker_names, str):
        marker_names = [marker_names]

    # Validate marker_names is a list
    if not isinstance(marker_names, list):
        raise ValueError(
            "marker_names must be a string or list of strings, "
            f"got {type(marker_names).__name__}"
        )

    # Validate all markers exist
    for marker_name in marker_names:
        if sum(adata.var_names == marker_name) == 0:
            raise ValueError(f"{marker_name} not found in var_names")

        if sum(adata.var_names == marker_name) > 1:
            raise ValueError(f"Multiple {marker_name} found in var_names")

    # Calculate grid dimensions
    n_markers = len(marker_names)
    if nrow is None:
        nrow = int(np.sqrt(n_markers))
    ncol = n_markers // nrow + (1 if n_markers % nrow != 0 else 0)

    # Downsample if requested
    if sample_size is not None and sample_size < adata.n_obs:
        adata = _downsample_cells(adata, sample_size=sample_size, seed=seed)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(figsize_sub[0] * ncol, figsize_sub[1] * nrow),
        squeeze=False,  # Always return 2D array for consistent indexing
    )
    axes_flat = axes.flatten()

    # Hide unused subplots
    for i in range(n_markers, len(axes_flat)):
        axes_flat[i].axis("off")

    # Plot each marker
    for i, marker_name in enumerate(marker_names):
        ax = axes_flat[i]

        # Extract marker values
        x = adata[:, marker_name].X.flatten()

        # Plot density
        sns.kdeplot(x=x, ax=ax)

        # Plot quantile lines
        for quantile, color in zip(
            quantiles, sns.color_palette(n_colors=len(quantiles))
        ):
            v_quantile = np.quantile(x, quantile)
            ax.axvline(
                v_quantile,
                color=color,
                linestyle="--",
                label=f"{quantile} ({v_quantile:.3g})",
                linewidth=linewidth,
            )

        ax.set_xlabel(marker_name)
        ax.set_ylabel("Density")

        if legend:
            ax.legend(title="Quantiles (Values)", loc="best", fontsize="small")

    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Return single axes for single marker, array for multiple markers
    if n_markers == 1:
        return fig, axes_flat[0]
    else:
        return fig, axes_flat[:n_markers]
