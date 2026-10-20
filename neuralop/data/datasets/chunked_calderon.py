"""Chunked boundary-data loader for Calderon coefficient-recovery experiments."""

from bisect import bisect_right
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..transforms.data_processors import DataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer
from .chunked_multioperator_darcy import (
    _discover_chunks,
    _load_chunk,
    _split_chunks,
)
from .multioperator_darcy import _prepare_fixed_context_indices


PathLike = Union[str, Path]
ChunkSelection = Sequence[Union[int, PathLike]]
_REQUIRED_KEYS = {"k", "f_boundary", "n_boundary"}


def _validate_calderon_chunk(chunk, path):
    """Validate one chunk and return its shape metadata.

    Expected fields are scalar 2D coefficients ``k`` with shape
    ``(n_operators, height, width)`` and paired boundary traces with shape
    ``(n_operators, n_examples, n_boundary_points)``.
    """

    if not isinstance(chunk, dict):
        raise TypeError(f"Expected {path} to contain a dictionary.")
    missing = _REQUIRED_KEYS - set(chunk)
    if missing:
        raise KeyError(f"Chunk {path} is missing required keys {missing}.")

    k = chunk["k"]
    f_boundary = chunk["f_boundary"]
    n_boundary = chunk["n_boundary"]
    if not all(torch.is_tensor(value) for value in (k, f_boundary, n_boundary)):
        raise TypeError(f"Expected k, f_boundary, and n_boundary in {path} to be tensors.")
    if k.ndim != 3:
        raise ValueError(
            f"Expected scalar 2D k in {path} to have shape "
            f"(n_operators, height, width), got {tuple(k.shape)}."
        )
    if f_boundary.ndim != 3 or n_boundary.ndim != 3:
        raise ValueError(
            f"Expected boundary traces in {path} to have shape "
            "(n_operators, n_examples, n_boundary_points), got "
            f"f_boundary={tuple(f_boundary.shape)} and "
            f"n_boundary={tuple(n_boundary.shape)}."
        )
    if f_boundary.shape != n_boundary.shape:
        raise ValueError(
            f"Expected paired boundary traces in {path} to have identical shapes, "
            f"got {tuple(f_boundary.shape)} and {tuple(n_boundary.shape)}."
        )
    if k.shape[0] != f_boundary.shape[0]:
        raise ValueError(
            f"Operator counts must agree in {path}; got k={k.shape[0]} and "
            f"boundary data={f_boundary.shape[0]}."
        )
    if any(size < 1 for size in (*k.shape, *f_boundary.shape[1:])):
        raise ValueError(f"Expected {path} to contain non-empty operators, grids, and traces.")

    return (
        int(k.shape[0]),
        tuple(k.shape[1:]),
        int(f_boundary.shape[1]),
        int(f_boundary.shape[2]),
    )


class ChunkedCalderonDataset(Dataset):
    """Calderon episodes backed by a one-chunk CPU cache.

    Each item selects one operator, samples a set of paired Dirichlet/Neumann
    traces for context, and uses that operator's 2D coefficient field as the
    target. No boundary example is reserved as a query.
    """

    def __init__(
        self,
        chunk_paths: Sequence[PathLike],
        n_context: int,
        n_samples: Optional[int] = None,
        random_context: bool = True,
        fixed_context_indices: Optional[Union[Sequence[int], torch.Tensor]] = None,
    ):
        super().__init__()
        if not chunk_paths:
            raise ValueError("chunk_paths must contain at least one path.")
        if n_context < 1:
            raise ValueError("n_context must be at least 1.")

        self.chunk_paths = [Path(path).expanduser().resolve() for path in chunk_paths]
        if len(set(self.chunk_paths)) != len(self.chunk_paths):
            raise ValueError("chunk_paths must not contain duplicates.")
        self.n_context = int(n_context)
        self.random_context = bool(random_context)
        self.chunk_sizes = []
        self.coefficient_shape = None
        self.n_examples = None
        self.n_boundary_points = None

        # Inspect chunks one at a time, retaining only shape metadata.
        for path in self.chunk_paths:
            chunk = _load_chunk(path)
            n_operators, coefficient_shape, n_examples, n_boundary_points = (
                _validate_calderon_chunk(chunk, path)
            )
            if self.coefficient_shape is None:
                self.coefficient_shape = coefficient_shape
                self.n_examples = n_examples
                self.n_boundary_points = n_boundary_points
            elif (
                coefficient_shape != self.coefficient_shape
                or n_examples != self.n_examples
                or n_boundary_points != self.n_boundary_points
            ):
                raise ValueError(f"Chunk {path} is incompatible with earlier chunks.")
            self.chunk_sizes.append(n_operators)
            del chunk

        if self.n_context > self.n_examples:
            raise ValueError(
                f"n_context={self.n_context} exceeds the {self.n_examples} paired "
                "boundary examples available per operator."
            )
        self.fixed_context_indices, _ = _prepare_fixed_context_indices(
            fixed_context_indices,
            n_context=self.n_context,
            n_pairs=self.n_examples,
            source="chunked Calderon dataset",
        )

        self.n_operators = sum(self.chunk_sizes)
        self.n_samples = self.n_operators if n_samples is None else int(n_samples)
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1.")

        self._operator_offsets = [0]
        for size in self.chunk_sizes:
            self._operator_offsets.append(self._operator_offsets[-1] + size)
        self._sample_counts = self._allocate_samples(self.n_samples)
        self._cached_chunk_index = None
        self._cached_chunk = None
        self.chunk_load_count = 0

    def _allocate_samples(self, total):
        counts = [(total * size) // self.n_operators for size in self.chunk_sizes]
        remainder = total - sum(counts)
        fractions = [
            ((total * size) % self.n_operators, -index, index)
            for index, size in enumerate(self.chunk_sizes)
        ]
        for _, _, index in sorted(fractions, reverse=True)[:remainder]:
            counts[index] += 1
        return counts

    @property
    def sample_counts(self):
        return tuple(self._sample_counts)

    @property
    def cached_chunk_index(self):
        return self._cached_chunk_index

    def clear_cache(self):
        self._cached_chunk_index = None
        self._cached_chunk = None

    def _get_chunk(self, chunk_index):
        if chunk_index != self._cached_chunk_index:
            chunk = _load_chunk(self.chunk_paths[chunk_index])
            _validate_calderon_chunk(chunk, self.chunk_paths[chunk_index])
            self._cached_chunk = chunk
            self._cached_chunk_index = chunk_index
            self.chunk_load_count += 1
        return self._cached_chunk

    def _global_to_local(self, index):
        operator_position = index % self.n_operators
        chunk_index = bisect_right(self._operator_offsets, operator_position) - 1
        local_operator = operator_position - self._operator_offsets[chunk_index]
        episode = index // self.n_operators
        local_episode = local_operator + episode * self.chunk_sizes[chunk_index]
        return chunk_index, local_episode

    def _context_indices(self, episode):
        if self.fixed_context_indices is not None:
            return self.fixed_context_indices
        if self.random_context:
            return torch.randperm(self.n_examples)[: self.n_context]
        if self.n_context == self.n_examples:
            return torch.arange(self.n_examples, dtype=torch.long)
        start = episode % self.n_examples
        return (torch.arange(self.n_context, dtype=torch.long) + start) % self.n_examples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if isinstance(index, tuple):
            chunk_index, local_episode = (int(value) for value in index)
        else:
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            chunk_index, local_episode = self._global_to_local(index)

        if chunk_index < 0 or chunk_index >= len(self.chunk_paths):
            raise IndexError(chunk_index)
        chunk = self._get_chunk(chunk_index)
        operator_index = local_episode % self.chunk_sizes[chunk_index]
        episode = local_episode // self.chunk_sizes[chunk_index]
        context_indices = self._context_indices(episode)

        return {
            "f_boundary_context": chunk["f_boundary"][
                operator_index, context_indices
            ].float().unsqueeze(1),
            "n_boundary_context": chunk["n_boundary"][
                operator_index, context_indices
            ].float().unsqueeze(1),
            "y": chunk["k"][operator_index].float().unsqueeze(0),
        }


class ChunkedCalderonBatchSampler(Sampler):
    """Batch episodes from one chunk at a time to reuse its CPU cache."""

    def __init__(self, dataset, batch_size, shuffle, drop_last=False, seed=0):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        chunk_indices = list(range(len(self.dataset.chunk_paths)))
        if self.shuffle:
            order = torch.randperm(len(chunk_indices), generator=generator).tolist()
            chunk_indices = [chunk_indices[index] for index in order]

        for chunk_index in chunk_indices:
            count = self.dataset.sample_counts[chunk_index]
            episodes = list(range(count))
            if self.shuffle:
                order = torch.randperm(count, generator=generator).tolist()
                episodes = [episodes[index] for index in order]
            batch = []
            for episode in episodes:
                batch.append((chunk_index, episode))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return sum(
                count // self.batch_size for count in self.dataset.sample_counts
            )
        return sum(
            (count + self.batch_size - 1) // self.batch_size
            for count in self.dataset.sample_counts
            if count
        )


def _fit_normalizers(chunk_paths, encode_f_boundary, encode_n_boundary, encode_k):
    enabled_fields = []
    if encode_f_boundary:
        enabled_fields.append("f_boundary")
    if encode_n_boundary:
        enabled_fields.append("n_boundary")
    if encode_k:
        enabled_fields.append("k")
    if not enabled_fields:
        return None, None, None

    stats = {
        field: {"count": 0, "mean": None, "m2": None}
        for field in enabled_fields
    }
    for path in chunk_paths:
        chunk = _load_chunk(path)
        _validate_calderon_chunk(chunk, path)
        for field in enabled_fields:
            values = chunk[field].float().to(torch.float64)
            batch_count = values.numel()
            batch_mean = values.mean()
            batch_m2 = (values - batch_mean).square().sum()
            field_stats = stats[field]
            if field_stats["count"] == 0:
                field_stats["mean"] = batch_mean
                field_stats["m2"] = batch_m2
            else:
                count = field_stats["count"]
                total = count + batch_count
                delta = batch_mean - field_stats["mean"]
                field_stats["mean"] += delta * batch_count / total
                field_stats["m2"] += (
                    batch_m2 + delta.square() * count * batch_count / total
                )
            field_stats["count"] += batch_count
        del chunk

    normalizers = {}
    for field, field_stats in stats.items():
        count = field_stats["count"]
        if count < 2:
            raise ValueError(f"At least two values are required to normalize {field}.")
        variance = field_stats["m2"] / (count - 1)
        normalizers[field] = UnitGaussianNormalizer(
            mean=field_stats["mean"].float().reshape(1, 1, 1, 1),
            std=variance.clamp_min(0).sqrt().float().reshape(1, 1, 1, 1),
            dim=[0, 2, 3],
        )
    return (
        normalizers.get("f_boundary"),
        normalizers.get("n_boundary"),
        normalizers.get("k"),
    )


class CalderonDataProcessor(DataProcessor):
    """Normalize paired boundary traces and a 2D coefficient target."""

    def __init__(
        self,
        f_boundary_normalizer=None,
        n_boundary_normalizer=None,
        coefficient_normalizer=None,
    ):
        super().__init__()
        self.f_boundary_normalizer = f_boundary_normalizer
        self.n_boundary_normalizer = n_boundary_normalizer
        self.coefficient_normalizer = coefficient_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        for normalizer in (
            self.f_boundary_normalizer,
            self.n_boundary_normalizer,
            self.coefficient_normalizer,
        ):
            if normalizer is not None:
                normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        del batched
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                data_dict[key] = value.to(self.device)

        if self.f_boundary_normalizer is not None:
            data_dict["f_boundary_context"] = self.f_boundary_normalizer.transform(
                data_dict["f_boundary_context"]
            )
        if self.n_boundary_normalizer is not None:
            data_dict["n_boundary_context"] = self.n_boundary_normalizer.transform(
                data_dict["n_boundary_context"]
            )
        if self.coefficient_normalizer is not None and self.training:
            data_dict["y"] = self.coefficient_normalizer.transform(data_dict["y"])
        return data_dict

    def postprocess(self, output, data_dict):
        if self.coefficient_normalizer is not None and not self.training:
            output = self.coefficient_normalizer.inverse_transform(output)
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict


def load_chunked_calderon(
    chunk_dir: PathLike,
    n_context: int,
    batch_size: int,
    test_batch_size: Optional[int] = None,
    n_train_samples: Optional[int] = None,
    n_val_samples: Optional[int] = None,
    n_test_samples: Optional[int] = None,
    *,
    n_train_chunks: Optional[int] = None,
    n_val_chunks: Optional[int] = None,
    n_test_chunks: Optional[int] = None,
    train_chunks: Optional[ChunkSelection] = None,
    val_chunks: Optional[ChunkSelection] = None,
    test_chunks: Optional[ChunkSelection] = None,
    chunk_pattern: str = "*chunk*.pt",
    encode_f_boundary: bool = True,
    encode_n_boundary: bool = True,
    encode_k: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
    fixed_context_indices: Optional[Union[Sequence[int], torch.Tensor]] = None,
):
    """Load chunked Calderon data with train/validation/test chunk splits.

    Each chunk must contain ``k`` shaped ``(operators, height, width)`` and
    paired ``f_boundary``/``n_boundary`` arrays shaped
    ``(operators, examples, boundary_points)``. Returned samples contain
    ``f_boundary_context``, ``n_boundary_context``, and ``y`` (the coefficient
    field). The return tuple is ``(train_loader, {"val": val_loader},
    test_loader, data_processor)``.
    """

    if num_workers != 0:
        raise ValueError(
            "Chunked Calderon loading currently requires num_workers=0 so the "
            "one-chunk cache is not duplicated across worker processes."
        )
    if test_batch_size is None:
        test_batch_size = batch_size

    resolved_dir = Path(chunk_dir).expanduser().resolve()
    discovered = _discover_chunks(resolved_dir, chunk_pattern)
    train_paths, val_paths, test_paths = _split_chunks(
        discovered,
        resolved_dir,
        train_chunks,
        val_chunks,
        test_chunks,
        n_train_chunks,
        n_val_chunks,
        n_test_chunks,
    )

    train_dataset = ChunkedCalderonDataset(
        train_paths,
        n_context=n_context,
        n_samples=n_train_samples,
        random_context=True,
        fixed_context_indices=fixed_context_indices,
    )
    val_dataset = ChunkedCalderonDataset(
        val_paths,
        n_context=n_context,
        n_samples=n_val_samples,
        random_context=False,
        fixed_context_indices=fixed_context_indices,
    )
    test_dataset = ChunkedCalderonDataset(
        test_paths,
        n_context=n_context,
        n_samples=n_test_samples,
        random_context=False,
        fixed_context_indices=fixed_context_indices,
    )

    f_normalizer, n_normalizer, k_normalizer = _fit_normalizers(
        train_paths, encode_f_boundary, encode_n_boundary, encode_k
    )
    data_processor = CalderonDataProcessor(
        f_boundary_normalizer=f_normalizer,
        n_boundary_normalizer=n_normalizer,
        coefficient_normalizer=k_normalizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ChunkedCalderonBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed
        ),
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ChunkedCalderonBatchSampler(
            val_dataset, batch_size=test_batch_size, shuffle=False, seed=seed
        ),
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=ChunkedCalderonBatchSampler(
            test_dataset, batch_size=test_batch_size, shuffle=False, seed=seed
        ),
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, {"val": val_loader}, test_loader, data_processor


def _normalizer_checkpoint(normalizer):
    if normalizer is None:
        return None
    return {
        "mean": normalizer.mean.detach().cpu(),
        "std": normalizer.std.detach().cpu(),
        "eps": normalizer.eps,
        "dim": normalizer.dim,
    }


def _normalizer_from_checkpoint(checkpoint):
    if checkpoint is None:
        return None
    return UnitGaussianNormalizer(
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        eps=checkpoint.get("eps", 1e-7),
        dim=checkpoint.get("dim"),
    )


def save_calderon_data_processor(data_processor, path, metadata=None):
    """Save Calderon normalization state for consistent later inference."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "f_boundary_normalizer": _normalizer_checkpoint(
                data_processor.f_boundary_normalizer
            ),
            "n_boundary_normalizer": _normalizer_checkpoint(
                data_processor.n_boundary_normalizer
            ),
            "coefficient_normalizer": _normalizer_checkpoint(
                data_processor.coefficient_normalizer
            ),
            "metadata": {} if metadata is None else metadata,
        },
        path,
    )


def load_calderon_data_processor(path, device="cpu"):
    """Restore saved Calderon normalization state and optional metadata."""

    checkpoint = torch.load(
        Path(path).as_posix(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Calderon data processor checkpoint format.")
    processor = CalderonDataProcessor(
        f_boundary_normalizer=_normalizer_from_checkpoint(
            checkpoint["f_boundary_normalizer"]
        ),
        n_boundary_normalizer=_normalizer_from_checkpoint(
            checkpoint["n_boundary_normalizer"]
        ),
        coefficient_normalizer=_normalizer_from_checkpoint(
            checkpoint["coefficient_normalizer"]
        ),
    ).to(device)
    return processor, checkpoint.get("metadata", {})
