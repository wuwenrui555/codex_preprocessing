# CODEX Preprocessing

Co-detection by indexing (CODEX) analysis: data preprocessing for downstream analysis.

## Installation

```bash
env_name=codex_preprocessing

# Remove existing environment if exists
conda deactivate && conda env remove -y -n $env_name

# Create and activate a new conda environment
mamba create -y -n $env_name python=3.10
mamba activate $env_name
```

```bash
pip install git+https://github.com/wuwenrui555/codex_preprocessing.git

pip install git+https://github.com/wuwenrui555/codex_preprocessing.git@dev
```

## Release Notes

- v0.1.0 (2025-12-06):
    - Add `preprocessing` module for data preprocessing
        - Modular functions for each preprocessing step:
            - Extreme value filtering
            - Nucleus signal normalization
            - Arcsinh transformation
            - Quantile normalization
        - Visualization tools to decide preprocessing parameters before each step
        - Downsampling strategy for efficient visualization
        - Comprehensive processing history tracking
    - Add `constants` module for storing constant values

- v0.1.1 (2025-12-06):
    - Update `ExtremeCutoff` class in `preprocessing` module: add options for not filtering lower or upper extreme values

- v0.1.2 (2026-01-07):
    - Fix wrong package name in `__init__.py`.

## Tutorials

- [Data Preprocessing](/notebook/data_preprocessing.ipynb)
