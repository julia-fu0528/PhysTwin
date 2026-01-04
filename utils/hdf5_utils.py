import h5py
from pathlib import Path
from typing import Any, Iterable, Optional

class H5Array:
    """
    Minimal zarr-like wrapper for a single HDF5 dataset stored at key 'data'.
    Keeps the underlying file open until `close` is called.
    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        shape: Optional[Iterable[int]] = None,
        chunks: Optional[Iterable[int]] = None,
        dtype: Any = None,
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
        dataset: str = "data",
        **kwargs
    ):
        self.path = str(path)
        self._file = None
        self._dset = None
        file_mode = "r" if mode == "r" else ("w" if mode == "w" else "a")
        self._file = h5py.File(self.path, file_mode, **kwargs)

        if dataset in self._file:
            self._dset = self._file[dataset]
        else:
            if shape is None or dtype is None:
                raise ValueError("shape and dtype are required when creating a new HDF5 dataset")
            self._dset = self._file.create_dataset(
                dataset,
                shape=tuple(shape),
                dtype=dtype,
                chunks=tuple(chunks) if chunks is not None else None,
                compression=compression,
                compression_opts=compression_level if compression else None,
            )

    def __getitem__(self, idx):
        return self._dset.__getitem__(idx)

    def __setitem__(self, idx, value):
        self._dset.__setitem__(idx, value)

    @property
    def shape(self):
        return self._dset.shape

    @property
    def dtype(self):
        return self._dset.dtype

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        try:
            if self._file:
                self._file.close()
        finally:
            self._file = None
            self._dset = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()