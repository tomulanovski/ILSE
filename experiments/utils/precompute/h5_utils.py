#!/usr/bin/env python3
"""
HDF5 utility functions for saving and loading precomputed embeddings.
"""
import h5py
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


def save_embeddings_to_h5(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    texts: List[str] = None,
    metadata: Dict[str, Any] = None,
    compression: str = "gzip",
    compression_opts: int = 4
):
    """
    Save embeddings, labels, and optionally texts to HDF5 file with compression and metadata.

    Args:
        embeddings: Array of shape (num_samples, num_layers, hidden_dim)
        labels: Array of shape (num_samples,)
        save_path: Path to save H5 file
        texts: List of original texts (optional, for traceability)
        metadata: Dictionary of metadata to store (task_name, pooling_method, etc.)
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_opts: Compression level (1-9 for gzip)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        # Store embeddings with compression
        f.create_dataset(
            'embeddings',
            data=embeddings.astype(np.float32),
            compression=compression,
            compression_opts=compression_opts if compression == 'gzip' else None,
            chunks=True  # Enable chunking for better partial reads
        )

        # Store labels (handle both int and string labels)
        if labels.dtype.kind in ('U', 'O', 'S'):  # String labels
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                'labels',
                data=np.array(labels, dtype=object),
                dtype=dt,
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )
        else:  # Numeric labels
            f.create_dataset(
                'labels',
                data=labels.astype(np.int64),
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )

        # Store texts if provided (for traceability)
        if texts is not None:
            # HDF5 requires special string dtype
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                'texts',
                data=np.array(texts, dtype=object),
                dtype=dt,
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )

        # Store metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    f.attrs[key] = json.dumps(value)
                else:
                    f.attrs[key] = value

        # Always store shape info
        f.attrs['num_samples'] = embeddings.shape[0]
        f.attrs['num_layers'] = embeddings.shape[1]
        f.attrs['hidden_dim'] = embeddings.shape[2]
        f.attrs['has_texts'] = texts is not None

    # Report file size
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {save_path.name}: {embeddings.shape} -> {file_size_mb:.1f} MB")


class ChunkedH5Writer:
    """
    Context manager for writing embeddings to HDF5 in chunks.
    Avoids loading all data into RAM at once.
    """

    def __init__(
        self,
        save_path: str,
        total_samples: int,
        num_layers: int,
        hidden_dim: int,
        labels: np.ndarray,
        texts: List[str] = None,
        metadata: Dict[str, Any] = None,
        compression: str = "gzip",
        compression_opts: int = 4
    ):
        """
        Initialize chunked writer.

        Args:
            save_path: Path to save H5 file
            total_samples: Total number of samples
            num_layers: Number of layers
            hidden_dim: Hidden dimension
            labels: Full labels array (must fit in RAM)
            texts: Full texts list (optional)
            metadata: Metadata dict
            compression: Compression algorithm
            compression_opts: Compression level
        """
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.total_samples = total_samples
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.labels = labels
        self.texts = texts
        self.metadata = metadata or {}
        self.compression = compression
        self.compression_opts = compression_opts

        self.h5_file = None
        self.embeddings_dataset = None
        self.current_idx = 0

    def __enter__(self):
        """Create HDF5 file and preallocate datasets."""
        self.h5_file = h5py.File(self.save_path, 'w')

        # Preallocate embeddings dataset with full shape
        self.embeddings_dataset = self.h5_file.create_dataset(
            'embeddings',
            shape=(self.total_samples, self.num_layers, self.hidden_dim),
            dtype=np.float32,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression == 'gzip' else None,
            chunks=True
        )

        # Store labels (assumed to fit in RAM, handle both int and string labels)
        if self.labels.dtype.kind in ('U', 'O', 'S'):  # String labels
            dt = h5py.special_dtype(vlen=str)
            self.h5_file.create_dataset(
                'labels',
                data=np.array(self.labels, dtype=object),
                dtype=dt,
                compression=self.compression,
                compression_opts=self.compression_opts if self.compression == 'gzip' else None
            )
        else:  # Numeric labels
            self.h5_file.create_dataset(
                'labels',
                data=self.labels.astype(np.int64),
                compression=self.compression,
                compression_opts=self.compression_opts if self.compression == 'gzip' else None
            )

        # Store texts if provided
        if self.texts is not None:
            dt = h5py.special_dtype(vlen=str)
            self.h5_file.create_dataset(
                'texts',
                data=np.array(self.texts, dtype=object),
                dtype=dt,
                compression=self.compression,
                compression_opts=self.compression_opts if self.compression == 'gzip' else None
            )

        # Store metadata
        for key, value in self.metadata.items():
            if isinstance(value, (list, dict)):
                self.h5_file.attrs[key] = json.dumps(value)
            else:
                self.h5_file.attrs[key] = value

        # Store shape info
        self.h5_file.attrs['num_samples'] = self.total_samples
        self.h5_file.attrs['num_layers'] = self.num_layers
        self.h5_file.attrs['hidden_dim'] = self.hidden_dim
        self.h5_file.attrs['has_texts'] = self.texts is not None

        return self

    def write_chunk(self, chunk_embeddings: np.ndarray):
        """
        Write a chunk of embeddings to the HDF5 file.

        Args:
            chunk_embeddings: Array of shape (chunk_size, num_layers, hidden_dim)
        """
        chunk_size = chunk_embeddings.shape[0]
        end_idx = self.current_idx + chunk_size

        if end_idx > self.total_samples:
            raise ValueError(f"Chunk would exceed total samples: {end_idx} > {self.total_samples}")

        # Write directly to HDF5 dataset slice
        self.embeddings_dataset[self.current_idx:end_idx] = chunk_embeddings.astype(np.float32)

        self.current_idx = end_idx

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file and report stats."""
        if self.h5_file is not None:
            self.h5_file.close()

            # Report file size on success
            if exc_type is None:
                file_size_mb = self.save_path.stat().st_size / (1024 * 1024)
                print(f"  Saved {self.save_path.name}: ({self.total_samples}, {self.num_layers}, {self.hidden_dim}) -> {file_size_mb:.1f} MB")


def load_embeddings_from_h5(
    load_path: str,
    load_labels: bool = True,
    load_texts: bool = False,
    load_metadata: bool = False
) -> tuple:
    """
    Load embeddings and optionally labels/texts/metadata from HDF5 file.

    Args:
        load_path: Path to H5 file
        load_labels: Whether to load labels
        load_texts: Whether to load original texts
        load_metadata: Whether to load metadata

    Returns:
        Tuple of (embeddings, labels, texts, metadata) depending on flags
        - If load_texts=False and load_metadata=False: (embeddings, labels)
        - If load_texts=True: adds texts to return
        - If load_metadata=True: adds metadata to return
    """
    with h5py.File(load_path, 'r') as f:
        # Try to read embeddings
        try:
            embeddings = np.asarray(f['embeddings'][:])  # Ensure proper numpy array
        except OSError as e:
            # Handle compression errors by reading in chunks
            print(f"  Warning: Error reading full compressed array: {e}")
            print(f"  Attempting chunk-based read as fallback...")

            dataset = f['embeddings']
            n_samples = dataset.shape[0]
            chunk_size = 1000  # Read 1000 samples at a time
            chunks = []

            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                try:
                    chunk = np.asarray(dataset[i:end])  # Ensure proper numpy array
                    chunks.append(chunk)
                    if (i // chunk_size + 1) % 5 == 0:
                        print(f"    Read {end}/{n_samples} samples...")
                except Exception as chunk_error:
                    raise RuntimeError(
                        f"Failed to read embeddings at samples {i}-{end}: {chunk_error}. "
                        f"File may be corrupted. Try re-running precompute with --force_recompute."
                    )

            embeddings = np.asarray(np.concatenate(chunks, axis=0))  # Ensure proper numpy array
            print(f"  ✓ Successfully read {n_samples} samples via chunked reading")

        labels = None
        if load_labels and 'labels' in f:
            try:
                labels = np.asarray(f['labels'][:])  # Ensure proper numpy array
            except OSError:
                # Chunk-based read for labels too
                print(f"  Warning: Reading labels in chunks...")
                dataset = f['labels']
                n_samples = dataset.shape[0]
                chunk_size = 5000
                chunks = []
                for i in range(0, n_samples, chunk_size):
                    end = min(i + chunk_size, n_samples)
                    chunks.append(np.asarray(dataset[i:end]))  # Ensure proper numpy array
                labels = np.asarray(np.concatenate(chunks, axis=0))  # Ensure proper numpy array

            # Handle string labels that come back as bytes
            if labels.dtype.kind in ('O', 'S'):
                labels = np.array([l.decode('utf-8') if isinstance(l, bytes) else l for l in labels])

        texts = None
        if load_texts and 'texts' in f:
            try:
                texts = [t.decode('utf-8') if isinstance(t, bytes) else t for t in f['texts'][:]]
            except OSError:
                # Chunk-based read for texts too
                print(f"  Warning: Reading texts in chunks...")
                dataset = f['texts']
                n_samples = dataset.shape[0]
                chunk_size = 5000
                all_texts = []
                for i in range(0, n_samples, chunk_size):
                    end = min(i + chunk_size, n_samples)
                    chunk_texts = dataset[i:end]
                    all_texts.extend([t.decode('utf-8') if isinstance(t, bytes) else t for t in chunk_texts])
                texts = all_texts

        metadata = None
        if load_metadata:
            metadata = dict(f.attrs)
            # Parse JSON strings back to objects
            for key, value in metadata.items():
                if isinstance(value, str) and value.startswith(('[', '{')):
                    try:
                        metadata[key] = json.loads(value)
                    except:
                        pass

    # Return based on what was requested
    result = [embeddings]
    if load_labels:
        result.append(labels)
    if load_texts:
        result.append(texts)
    if load_metadata:
        result.append(metadata)

    if len(result) == 1:
        return result[0]
    return tuple(result)


def get_h5_metadata(load_path: str) -> Dict[str, Any]:
    """Get metadata from H5 file without loading embeddings."""
    with h5py.File(load_path, 'r') as f:
        metadata = dict(f.attrs)
        # Parse JSON strings
        for key, value in metadata.items():
            if isinstance(value, str) and value.startswith(('[', '{')):
                try:
                    metadata[key] = json.loads(value)
                except:
                    pass
    return metadata


def validate_h5_file(load_path: str, expected_metadata: Dict[str, Any] = None) -> bool:
    """
    Validate that H5 file exists and optionally matches expected metadata.

    Args:
        load_path: Path to H5 file
        expected_metadata: Dict of metadata that must match

    Returns:
        True if valid, False otherwise
    """
    load_path = Path(load_path)
    if not load_path.exists():
        return False

    try:
        metadata = get_h5_metadata(load_path)

        # Check required keys exist
        required = ['num_samples', 'num_layers', 'hidden_dim']
        for key in required:
            if key not in metadata:
                print(f"  Warning: {load_path} missing '{key}' in metadata")
                return False

        # Check expected metadata matches
        if expected_metadata:
            for key, expected_value in expected_metadata.items():
                if key in metadata and metadata[key] != expected_value:
                    print(f"  Warning: {load_path} has {key}={metadata[key]}, expected {expected_value}")
                    return False

        return True
    except Exception as e:
        print(f"  Warning: Error validating {load_path}: {e}")
        return False


class PrecomputedEmbeddingDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that loads precomputed embeddings from HDF5.
    Supports lazy loading for memory efficiency.
    """

    def __init__(self, h5_path: str, lazy_load: bool = False):
        """
        Args:
            h5_path: Path to H5 file
            lazy_load: If True, load samples on-demand (slower but memory efficient)
                      If False, load all into RAM (faster but needs more memory)
        """
        self.h5_path = h5_path
        self.lazy_load = lazy_load

        if lazy_load:
            # Open file handle for lazy loading
            self.h5_file = h5py.File(h5_path, 'r')
            self.embeddings = self.h5_file['embeddings']
            self.labels = self.h5_file['labels']
            self.num_samples = self.embeddings.shape[0]
        else:
            # Load everything into RAM
            self.embeddings, self.labels = load_embeddings_from_h5(h5_path)
            self.num_samples = len(self.embeddings)
            self.h5_file = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            embeddings: (num_layers, hidden_dim) float32
            label: int64
        """
        emb = self.embeddings[idx]
        lbl = self.labels[idx]

        # Always ensure proper numpy array (for both lazy and non-lazy modes)
        emb = np.asarray(emb)
        lbl = int(lbl) if not isinstance(lbl, int) else lbl

        return torch.from_numpy(emb).float(), torch.tensor(lbl, dtype=torch.long)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the H5 file."""
        return get_h5_metadata(self.h5_path)

    @property
    def num_layers(self) -> int:
        return self.embeddings.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.embeddings.shape[2]
