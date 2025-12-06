# PyCODEX

Co-detection by indexing (CODEX) analysis using Python

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
