import importlib
import inspect
import os
import pickle
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch

_TORCH_LOAD = getattr(torch, "load")

_ALLOWED_PICKLE_GLOBALS = {
    ("builtins", "bytearray"): bytearray,
    ("builtins", "bytes"): bytes,
    ("builtins", "complex"): complex,
    ("builtins", "dict"): dict,
    ("builtins", "float"): float,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "int"): int,
    ("builtins", "list"): list,
    ("builtins", "range"): range,
    ("builtins", "set"): set,
    ("builtins", "slice"): slice,
    ("builtins", "str"): str,
    ("builtins", "tuple"): tuple,
    ("collections", "OrderedDict"): OrderedDict,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
}

_ALLOWED_NUMPY_GLOBALS = {
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
}


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only simple builtins and NumPy array reconstruction helpers."""

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key in _ALLOWED_PICKLE_GLOBALS:
            return _ALLOWED_PICKLE_GLOBALS[key]
        if key in _ALLOWED_NUMPY_GLOBALS:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is not allowed")


def load_wesad_pickle(path: str) -> Any:
    """Load trusted WESAD exports without allowing arbitrary pickle globals."""
    with open(path, "rb") as handle:
        return RestrictedNumpyUnpickler(handle).load()


def load_npz_archive(path: str) -> np.lib.npyio.NpzFile:
    """Open an `.npz` archive without allowing pickled object arrays."""
    return np.load(path, allow_pickle=False)


def load_torch_checkpoint(
    path: str,
    map_location: Optional[Any] = None,
) -> Any:
    """Load a tensor-only PyTorch checkpoint."""
    if "weights_only" not in inspect.signature(_TORCH_LOAD).parameters:
        if os.environ.get("PULSE_ALLOW_UNSAFE_TORCH_LOAD") == "1":
            return _TORCH_LOAD(path, map_location=map_location)
        raise RuntimeError(
            "Safe checkpoint loading requires a PyTorch build that supports "
            "weights_only=True. Upgrade PyTorch or set "
            "PULSE_ALLOW_UNSAFE_TORCH_LOAD=1 for a trusted legacy checkpoint."
        )
    return _TORCH_LOAD(path, map_location=map_location, weights_only=True)
