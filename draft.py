# %%
import re
from importlib import reload
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if True:
    import pycodex2
    import pycodex2.preprocessing

    reload(pycodex2)
    reload(pycodex2.preprocessing)
    from pycodex2.preprocessing import (
        ExtremeCutoff,
        _downsample_cells,
        arcsinh_transformation,
        combine_csv_data,
        nucleus_signal_normalization,
        plot_arcsinh_transformation,
        plot_quantile_normalization,
        print_processing_history,
        quantile_normalization,
    )


MARKER_NAMES = [
    "DAPI",
    "CD45",
    "CD3e",
    "CD8",
    "CD4",
    "CD45RO",
    "CD45RA",
    "CD69",
    "CD57",
    "CD56",
    "FoxP3",
    "CD28",
    "CD86",
    "T-bet",
    "TCF1_7",
    "IFN-y",
    "GranzymeB",
    "Tox_Tox2",
    "Tim-3",
    "PD-1",
    "LAG-3",
    "CD20",
    "CD138",
    "TREM2",
    # "CD68",
    "CD163",
    "CD16",
    "CD11b",
    "CD11c",
    "C1Q",
    "MPO",
    "IDO-1",
    "PDL1",
    "CA9",
    "Cytokeratin",
    "HLA1",
    "Ki-67",
    "P53",
    "CD31",
    "Podoplanin",
    "aSMA",
    "NaKATP",
    "VDAC1",
    "ATP5A",
    "GLUT1",
    "G6PD",
]

# %%
adata_f = Path("combined_data.h5ad")

if adata_f.exists():
    adata = ad.read_h5ad(adata_f)
else:
    root_dir = Path(
        "/mnt/nfs/home/wenruiwu/projects/bidmc-jiang-rcc/output/data/20250116_ometiff/"
    )
    region_dirs = list(root_dir.glob("*RCC*"))

    data_fs = [
        region_dir / "20250125_whole_cell" / "dataScaleSize.csv"
        for region_dir in region_dirs
    ]
    data_ids = [region_dir.name for region_dir in region_dirs]
    marker_names = MARKER_NAMES
    rename_dict = {
        "cellLabel": "cell_id",
        "cellSize": "cell_size",
        "X_cent": "x_cent",
        "Y_cent": "y_cent",
    }
    col_cell_id = "cell_id"
    col_data_id = "core_id"

    adata = combine_csv_data(
        data_fs=data_fs,
        data_ids=data_ids,
        marker_names=marker_names,
        rename_dict=rename_dict,
        col_cell_id=col_cell_id,
        col_data_id=col_data_id,
    )
    adata.obs["tma_id"] = [
        re.search(r"(TMA\d+)", i).group(1) for i in adata.obs["core_id"]
    ]
    print(adata)
    adata.write_h5ad("combined_data.h5ad")

# %%
extreme_cutoff = ExtremeCutoff(values=adata.obs["cell_size"])
print(extreme_cutoff)

extreme_cutoff.test_cutoffs(n_sigmas=list(range(1, 6)))
extreme_cutoff.plot_cutoffs()

# mask = extreme_cutoff.filter_values(method="median", n_sigma_lower=2.5, n_sigma_upper=5)
mask_size = extreme_cutoff.filter_values(method="median", n_sigma=3)
print(f"Filtering {np.sum(~mask_size):,} cells based on cell_size")

mask_dapi = (adata[:, "DAPI"].X > 0).flatten()
print(f"Filtering {np.sum(~mask_dapi):,} cells with zero DAPI signal")

adata_filtered = adata[mask_size & mask_dapi].copy()
adata_filtered.layers["scale_size"] = adata_filtered.X.copy()

print(adata_filtered)

# %%
nucleus_signal_normalization(
    adata_filtered, col_data_id="core_id", marker_nucleus="DAPI", inplace=True
)

print(adata_filtered)
print_processing_history(adata_filtered)

# %%
# Downsample cells for visualization
adata_sm = _downsample_cells(adata_filtered, sample_size=100000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
sns.kdeplot(
    x=adata_sm[:, "DAPI"].layers["scale_size"].flatten(),
    hue=adata_sm.obs["tma_id"],
    log_scale=(True, False),
    legend=True,
    ax=ax,
)
ax.set_title("DAPI Before Normalization")

ax = axes[1]
sns.kdeplot(
    x=adata_sm[:, "DAPI"].X.flatten(),
    hue=adata_sm.obs["tma_id"],
    log_scale=(True, False),
    legend=True,
    ax=ax,
)
ax.set_title("DAPI After Normalization")

plt.show()

# %%
fig, axes = plot_arcsinh_transformation(
    adata_filtered,
    "CD3e",
    hue="tma_id",
    nrow=4,
    sample_size=100000,
    show=True,
)

# %%
arcsinh_transformation(adata_filtered, cofactor=0.01, inplace=True)

print(adata_filtered)
print_processing_history(adata_filtered)

# %%
fig, axes = plot_quantile_normalization(
    adata_filtered,
    ["CD3e", "CD45", "CD163", "DAPI"],
    sample_size=100000,
)

# %%
quantile_normalization(
    adata_filtered,
    min_quantile=0.01,
    max_quantile=0.999,
    inplace=True,
)

print(adata_filtered)
print_processing_history(adata_filtered)
