# %%
from pathlib import Path
from typing import Union

import anndata as ad
import pandas as pd
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
