from importlib.metadata import version
from time import gmtime, strftime

__version__ = version("codex_preprocessing")
print(f"Running codex_preprocessing (version={__version__})")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
