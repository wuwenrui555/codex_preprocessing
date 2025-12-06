from importlib.metadata import version
from time import gmtime, strftime

__version__ = version("pycodex2")
print(f"(Running pyCODEX2 {__version__})")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
