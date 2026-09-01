"""Memory-bounded loading for multi-operator Darcy datasets stored in chunks."""

from bisect import bisect_right
from pathlib import Path
import re
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..transforms.normalizers import UnitGaussianNormalizer
from .multioperator_darcy import (
    MultiOperatorDarcyDataProcessor,
    _validate_multioperator_tensors,
)


PathLike = Union[str, Path]
ChunkSelection = Sequence[Union[int, PathLike]]
_REQUIRED_KEYS = {"k", "f", "u"}


def _natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _load_chunk(path):
    # User-generated chunks may contain arbitrary optional metadata.
    return torch.load(Path(path).as_posix(), map_location="cpu", weights_only=False)


def _validate_chunk(chunk, path):
    if not isinstance(chunk, dict):
        raise TypeError(f"Expected {path} to contain a dictionary.")
    missing = _REQUIRED_KEYS - set(chunk)
    if missing:
        raise KeyError(f"Chunk {path} is missing required keys {missing}.")

    return _validate_multioperator_tensors(
        chunk["k"], chunk["f"], chunk["u"], source=f"chunk {path}"
    )


def _discover_chunks(chunk_dir, chunk_pattern):
    chunk_dir = Path(chunk_dir).expanduser().resolve()
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")
    paths = [
        path.resolve()
        for path in sorted(chunk_dir.glob(chunk_pattern), key=_natural_sort_key)
        if path.is_file() and path.name != "manifest.pt"
    ]
    if not paths:
        raise FileNotFoundError(
            f"No chunk files matching {chunk_pattern!r} were found in {chunk_dir}."
        )
    return paths


def _resolve_selection(selection, discovered, chunk_dir, name):
    discovered_by_path = {path.resolve(): path for path in discovered}
    resolved = []
    for item in selection:
        if isinstance(item, int):
            try:
                path = discovered[item]
            except IndexError as error:
                raise IndexError(
                    f"{name} index {item} is outside [0, {len(discovered) - 1}]."
                ) from error
        else:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = chunk_dir / path
            path = path.resolve()
            if path not in discovered_by_path:
                raise ValueError(f"{name} contains an undiscovered chunk: {path}")
            path = discovered_by_path[path]
        resolved.append(path)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} contains duplicate chunks.")
    return resolved


def _split_chunks(
    discovered,
    chunk_dir,
    train_chunks,
    val_chunks,
    test_chunks,
    n_train_chunks,
    n_val_chunks,
    n_test_chunks,
):
    explicit = (train_chunks, val_chunks, test_chunks)
    counts = (n_train_chunks, n_val_chunks, n_test_chunks)
    if any(selection is not None for selection in explicit):
        if not all(selection is not None for selection in explicit):
            raise ValueError(
                "Provide train_chunks, val_chunks, and test_chunks together."
            )
        if any(count is not None for count in counts):
            raise ValueError("Use either explicit selections or chunk counts, not both.")
        splits = (
            _resolve_selection(train_chunks, discovered, chunk_dir, "train_chunks"),
            _resolve_selection(val_chunks, discovered, chunk_dir, "val_chunks"),
            _resolve_selection(test_chunks, discovered, chunk_dir, "test_chunks"),
        )
    else:
        if n_train_chunks is None or n_val_chunks is None:
            raise ValueError(
                "Count-based splitting requires n_train_chunks and n_val_chunks."
            )
        if n_test_chunks is None:
            n_test_chunks = len(discovered) - n_train_chunks - n_val_chunks
        if min(n_train_chunks, n_val_chunks, n_test_chunks) <= 0:
            raise ValueError("Every split must contain at least one chunk.")
        requested = n_train_chunks + n_val_chunks + n_test_chunks
        if requested > len(discovered):
            raise ValueError(
                f"Requested {requested} chunks, but only {len(discovered)} were found."
            )
        train_end = n_train_chunks
        val_end = train_end + n_val_chunks
        test_end = val_end + n_test_chunks
        splits = (
            discovered[:train_end],
            discovered[train_end:val_end],
            discovered[val_end:test_end],
        )

    train, val, test = (list(split) for split in splits)
    if not train or not val or not test:
        raise ValueError("Every split must contain at least one chunk.")
    if (set(train) & set(val)) or (set(train) & set(test)) or (set(val) & set(test)):
        raise ValueError("Train, validation, and test chunks must not overlap.")
    return train, val, test


class ChunkedMultiOperatorDarcyDataset(Dataset):
    """Episodic Darcy dataset backed by a one-chunk CPU cache.

    Integer indices support ordinary random access. ChunkAwareBatchSampler
    supplies (chunk, episode) indices so consecutive batches use one chunk.
    """

    def __init__(
        self,
        chunk_paths: Sequence[PathLike],
        n_context: int,
        n_samples: Optional[Union[int, str]] = None,
        random_context: bool = True,
        include_k: bool = False,
        operator_direction: str = "f_to_u",
    ):
        super().__init__()
        if not chunk_paths:
            raise ValueError("chunk_paths must contain at least one path.")
        if operator_direction not in {"f_to_u", "u_to_f"}:
            raise ValueError(
                "operator_direction must be one of {'f_to_u', 'u_to_f'}."
            )
        if n_context < 1:
            raise ValueError("n_context must be at least 1.")

        self.chunk_paths = [
            Path(path).expanduser().resolve() for path in chunk_paths
        ]
        if len(set(self.chunk_paths)) != len(self.chunk_paths):
            raise ValueError("chunk_paths must not contain duplicates.")
        self.n_context = int(n_context)
        self.random_context = bool(random_context)
        self.include_k = bool(include_k)
        self.operator_direction = operator_direction
        self.chunk_sizes = []
        self.n_pairs = None
        self.spatial_shape = None
        self.coefficient_shape = None

        # Initialization examines one chunk at a time and retains only metadata.
        for path in self.chunk_paths:
            chunk = _load_chunk(path)
            n_operators, n_pairs, spatial_shape, coefficient_shape = _validate_chunk(
                chunk, path
            )
            if self.n_pairs is None:
                self.n_pairs = n_pairs
                self.spatial_shape = spatial_shape
                self.coefficient_shape = coefficient_shape
            elif (
                n_pairs != self.n_pairs
                or spatial_shape != self.spatial_shape
                or coefficient_shape != self.coefficient_shape
            ):
                raise ValueError(f"Chunk {path} is incompatible with earlier chunks.")
            self.chunk_sizes.append(n_operators)
            del chunk

        if self.n_context >= self.n_pairs:
            raise ValueError(
                f"Need at least n_context + 1 pairs, got n_context={self.n_context} "
                f"and n_pairs={self.n_pairs}."
            )

        self.n_operators = sum(self.chunk_sizes)
        if n_samples == "all_queries":
            self.n_samples = self.n_operators * self.n_pairs
        elif n_samples is None:
            self.n_samples = self.n_operators
        else:
            self.n_samples = int(n_samples)
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
        if not self.random_context:
            full_cycles, remainder = divmod(total, self.n_operators)
            counts = [full_cycles * size for size in self.chunk_sizes]
            for index, (start, end) in enumerate(
                zip(self._operator_offsets[:-1], self._operator_offsets[1:])
            ):
                counts[index] += max(0, min(remainder, end) - start)
            return counts

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
            _validate_chunk(chunk, self.chunk_paths[chunk_index])
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

    def _sample_pair_indices(self, local_episode, n_operators):
        operator_index = local_episode % n_operators
        if self.random_context:
            permutation = torch.randperm(self.n_pairs)
            query_index = int(permutation[0])
            context_indices = permutation[1 : self.n_context + 1]
        else:
            query_index = (local_episode // n_operators) % self.n_pairs
            available = torch.cat(
                (
                    torch.arange(query_index, dtype=torch.long),
                    torch.arange(query_index + 1, self.n_pairs, dtype=torch.long),
                )
            )
            context_indices = available[: self.n_context]
        return operator_index, context_indices, query_index

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
        operator_index, context_indices, query_index = self._sample_pair_indices(
            local_episode, self.chunk_sizes[chunk_index]
        )

        if self.operator_direction == "f_to_u":
            input_functions, output_functions = chunk["f"], chunk["u"]
        else:
            input_functions, output_functions = chunk["u"], chunk["f"]
        sample = {
            "u_context": input_functions[
                operator_index, context_indices
            ].float().unsqueeze(1),
            "f_context": output_functions[
                operator_index, context_indices
            ].float().unsqueeze(1),
            "u_query": input_functions[
                operator_index, query_index
            ].float().unsqueeze(0),
            "y": output_functions[operator_index, query_index].float().unsqueeze(0),
        }
        if self.include_k:
            sample["k"] = chunk["k"][operator_index].float().unsqueeze(0)
        return sample


class ChunkAwareBatchSampler(Sampler):
    """Group batches by chunk and optionally shuffle chunks and episodes."""

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
            permutation = torch.randperm(
                len(chunk_indices), generator=generator
            ).tolist()
            chunk_indices = [chunk_indices[index] for index in permutation]

        for chunk_index in chunk_indices:
            count = self.dataset.sample_counts[chunk_index]
            if self.shuffle:
                size = self.dataset.chunk_sizes[chunk_index]
                full_cycles, remainder = divmod(count, size)
                episodes = []
                for cycle in range(full_cycles):
                    permutation = torch.randperm(size, generator=generator)
                    episodes.extend((permutation + cycle * size).tolist())
                if remainder:
                    permutation = torch.randperm(size, generator=generator)[:remainder]
                    episodes.extend((permutation + full_cycles * size).tolist())
            else:
                episodes = range(count)
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


def _fit_normalizers(
    chunk_paths, input_field, output_field, encode_input, encode_output
):
    fields = []
    if encode_input:
        fields.append(input_field)
    if encode_output and output_field not in fields:
        fields.append(output_field)
    if not fields:
        return None, None

    stats = {
        field: {"mean": None, "m2": None, "count": 0, "spatial_ndim": 0}
        for field in fields
    }
    for path in chunk_paths:
        chunk = _load_chunk(path)
        _validate_chunk(chunk, path)
        for field in fields:
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
            field_stats["spatial_ndim"] = values.ndim - 2
        del chunk

    normalizers = {}
    for field, field_stats in stats.items():
        count = field_stats["count"]
        if count < 2:
            raise ValueError("At least two values are required to fit a normalizer.")
        mean = field_stats["mean"]
        variance = field_stats["m2"] / (count - 1)
        spatial_ndim = field_stats["spatial_ndim"]
        stat_shape = (1, 1) + (1,) * spatial_ndim
        normalizers[field] = UnitGaussianNormalizer(
            mean=mean.float().reshape(stat_shape),
            std=variance.clamp_min(0).sqrt().float().reshape(stat_shape),
            dim=[0] + list(range(2, spatial_ndim + 2)),
        )
    return (
        normalizers.get(input_field) if encode_input else None,
        normalizers.get(output_field) if encode_output else None,
    )


def load_chunked_multiop_darcy(
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
    encode_input: bool = True,
    encode_output: bool = True,
    include_k: bool = False,
    operator_direction: str = "f_to_u",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
):
    """Load chunked Darcy data with explicit train/validation/test splits.

    Splits may use contiguous counts in sorted filename order or three explicit
    selections. Explicit integers index the sorted files; names and paths work
    as well.

    Returns the train loader, a validation dictionary keyed by val, the final
    test loader, and the data processor.
    """
    if num_workers != 0:
        raise ValueError(
            "Chunked loading currently requires num_workers=0 so the one-chunk "
            "cache is not duplicated across worker processes."
        )
    if operator_direction not in {"f_to_u", "u_to_f"}:
        raise ValueError(
            "operator_direction must be one of {'f_to_u', 'u_to_f'}."
        )
    if test_batch_size is None:
        test_batch_size = batch_size

    chunk_dir = Path(chunk_dir).expanduser().resolve()
    discovered = _discover_chunks(chunk_dir, chunk_pattern)
    train_paths, val_paths, test_paths = _split_chunks(
        discovered,
        chunk_dir,
        train_chunks,
        val_chunks,
        test_chunks,
        n_train_chunks,
        n_val_chunks,
        n_test_chunks,
    )
    train_dataset = ChunkedMultiOperatorDarcyDataset(
        train_paths,
        n_context=n_context,
        n_samples=n_train_samples,
        random_context=True,
        include_k=include_k,
        operator_direction=operator_direction,
    )
    val_dataset = ChunkedMultiOperatorDarcyDataset(
        val_paths,
        n_context=n_context,
        n_samples="all_queries" if n_val_samples is None else n_val_samples,
        random_context=False,
        include_k=include_k,
        operator_direction=operator_direction,
    )
    test_dataset = ChunkedMultiOperatorDarcyDataset(
        test_paths,
        n_context=n_context,
        n_samples="all_queries" if n_test_samples is None else n_test_samples,
        random_context=False,
        include_k=include_k,
        operator_direction=operator_direction,
    )

    input_field, output_field = (
        ("f", "u") if operator_direction == "f_to_u" else ("u", "f")
    )
    in_normalizer, out_normalizer = _fit_normalizers(
        train_paths,
        input_field,
        output_field,
        encode_input,
        encode_output,
    )
    data_processor = MultiOperatorDarcyDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed
        ),
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            val_dataset, batch_size=test_batch_size, shuffle=False, seed=seed
        ),
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=ChunkAwareBatchSampler(
            test_dataset, batch_size=test_batch_size, shuffle=False, seed=seed
        ),
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, {"val": val_loader}, test_loader, data_processor
