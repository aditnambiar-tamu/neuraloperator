"""Data utilities for training FNOSetsInverse on Darcy operator datasets.

The inverse processor can consume the forward loaders directly because it
replaces their query target with ``k``. The loader wrapper remains available
when callers need batches whose ``y`` value is already the coefficient field.
"""

from pathlib import Path
from typing import Mapping, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..transforms.data_processors import DataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer
from .chunked_multioperator_darcy import (
    ChunkedMultiOperatorDarcyDataset,
    _load_chunk,
    _validate_chunk,
)
from .multioperator_darcy import MultiOperatorDarcyDataset


class CoefficientTargetDataset(Dataset):
    """Use the coefficient field ``k`` as an episodic dataset's target."""

    def __init__(self, dataset):
        if not getattr(dataset, "include_k", False):
            raise ValueError(
                "CoefficientTargetDataset requires a dataset created with "
                "include_k=True."
            )
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["y"] = sample["k"]
        return sample


class FNOSetsInverseDataProcessor(DataProcessor):
    """Normalize context functions and scalar coefficient targets."""

    def __init__(
        self,
        input_normalizer=None,
        output_normalizer=None,
        coefficient_normalizer=None,
    ):
        super().__init__()
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.coefficient_normalizer = coefficient_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.input_normalizer is not None:
            self.input_normalizer = self.input_normalizer.to(device)
        if self.output_normalizer is not None:
            self.output_normalizer = self.output_normalizer.to(device)
        if self.coefficient_normalizer is not None:
            self.coefficient_normalizer = self.coefficient_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                data_dict[key] = value.to(self.device)

        if "k" not in data_dict:
            raise KeyError(
                "FNOSetsInverseDataProcessor requires batches containing k. "
                "Create the dataset with include_k=True."
            )
        data_dict["y"] = data_dict["k"]

        if self.input_normalizer is not None:
            data_dict["u_context"] = self.input_normalizer.transform(
                data_dict["u_context"]
            )
        if self.output_normalizer is not None:
            data_dict["f_context"] = self.output_normalizer.transform(
                data_dict["f_context"]
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


def _update_scalar_stats(values, count, mean, m2):
    values = values.to(dtype=torch.float64)
    batch_count = values.numel()
    batch_mean = values.mean()
    batch_m2 = (values - batch_mean).square().sum()

    if count == 0:
        return batch_count, batch_mean, batch_m2

    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * batch_count / total
    m2 = m2 + batch_m2 + delta.square() * count * batch_count / total
    return total, mean, m2


def _validate_scalar_coefficients(dataset):
    if isinstance(dataset, ChunkedMultiOperatorDarcyDataset):
        coefficient_shape = dataset.coefficient_shape
        spatial_ndim = len(dataset.spatial_shape)
    elif isinstance(dataset, MultiOperatorDarcyDataset):
        spatial_ndim = dataset.f.ndim - 2
        coefficient_shape = tuple(dataset.k.shape[spatial_ndim + 1 :])
    else:
        raise TypeError(
            "Expected a MultiOperatorDarcyDataset or "
            "ChunkedMultiOperatorDarcyDataset."
        )

    if coefficient_shape:
        raise ValueError(
            "FNOSetsInverse scalar coefficient utilities require scalar k fields; "
            f"got trailing coefficient shape {coefficient_shape}."
        )
    return spatial_ndim


def fit_coefficient_normalizer(dataset):
    """Fit scalar ``k`` statistics using only the dataset's operators.

    Chunked datasets are processed one file at a time, so fitting the
    normalizer does not materialize the complete coefficient dataset in memory.
    Each operator contributes exactly once, independently of ``n_samples``.
    """

    if isinstance(dataset, CoefficientTargetDataset):
        dataset = dataset.dataset

    spatial_ndim = _validate_scalar_coefficients(dataset)
    count, mean, m2 = 0, None, None

    if isinstance(dataset, ChunkedMultiOperatorDarcyDataset):
        for path in dataset.chunk_paths:
            chunk = _load_chunk(path)
            _validate_chunk(chunk, path)
            count, mean, m2 = _update_scalar_stats(
                chunk["k"], count, mean, m2
            )
            del chunk
    else:
        coefficient = dataset.k[dataset.operator_indices]
        count, mean, m2 = _update_scalar_stats(coefficient, count, mean, m2)

    if count < 2:
        raise ValueError(
            "At least two coefficient values are required to normalize k."
        )

    stat_shape = (1, 1) + (1,) * spatial_ndim
    return UnitGaussianNormalizer(
        mean=mean.float().reshape(stat_shape),
        std=(m2 / (count - 1)).clamp_min(0).sqrt().float().reshape(stat_shape),
        dim=[0] + list(range(2, spatial_ndim + 2)),
    )


def wrap_fnosets_inverse_loader(loader):
    """Return a loader whose target is ``k`` while preserving its sampler.

    In particular, this retains ``ChunkAwareBatchSampler`` so a wrapped
    chunked loader still reads each chunk only once per epoch.
    """

    if isinstance(loader.dataset, CoefficientTargetDataset):
        return loader

    kwargs = {
        "batch_sampler": loader.batch_sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
    }
    generator = getattr(loader, "generator", None)
    if generator is not None:
        kwargs["generator"] = generator
    if getattr(loader, "pin_memory_device", ""):
        kwargs["pin_memory_device"] = loader.pin_memory_device
    if loader.num_workers:
        kwargs["persistent_workers"] = loader.persistent_workers
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor
        if loader.multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = loader.multiprocessing_context

    return DataLoader(CoefficientTargetDataset(loader.dataset), **kwargs)


def build_fnosets_inverse_data_processor(
    base_data_processor,
    training_dataset,
    coefficient_normalizer: Optional[UnitGaussianNormalizer] = None,
):
    """Build an inverse processor from a forward loader's normalizers."""

    base_dataset = (
        training_dataset.dataset
        if isinstance(training_dataset, CoefficientTargetDataset)
        else training_dataset
    )
    if not getattr(base_dataset, "include_k", False):
        raise ValueError(
            "FNOSetsInverse training requires a dataset created with "
            "include_k=True."
        )
    if coefficient_normalizer is None:
        coefficient_normalizer = fit_coefficient_normalizer(training_dataset)
    return FNOSetsInverseDataProcessor(
        input_normalizer=base_data_processor.in_normalizer,
        output_normalizer=base_data_processor.out_normalizer,
        coefficient_normalizer=coefficient_normalizer,
    )


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
        eps=checkpoint["eps"],
        dim=checkpoint["dim"],
    )


def save_fnosets_inverse_data_processor(
    data_processor, path, metadata: Optional[dict] = None
):
    """Save inverse normalization statistics for standalone evaluation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_normalizer": _normalizer_checkpoint(
                data_processor.input_normalizer
            ),
            "output_normalizer": _normalizer_checkpoint(
                data_processor.output_normalizer
            ),
            "coefficient_normalizer": _normalizer_checkpoint(
                data_processor.coefficient_normalizer
            ),
            "metadata": {} if metadata is None else metadata,
        },
        path,
    )


def load_fnosets_inverse_data_processor(path, device="cpu"):
    """Load inverse normalization statistics saved during training."""

    checkpoint = torch.load(
        Path(path).as_posix(), map_location="cpu", weights_only=False
    )
    data_processor = FNOSetsInverseDataProcessor(
        input_normalizer=_normalizer_from_checkpoint(
            checkpoint["input_normalizer"]
        ),
        output_normalizer=_normalizer_from_checkpoint(
            checkpoint["output_normalizer"]
        ),
        coefficient_normalizer=_normalizer_from_checkpoint(
            checkpoint["coefficient_normalizer"]
        ),
    ).to(device)
    return data_processor, checkpoint.get("metadata", {})


def wrap_fnosets_inverse_loaders(loaders: Mapping[str, DataLoader]):
    """Wrap every loader in a named validation/test loader dictionary."""

    return {
        name: wrap_fnosets_inverse_loader(loader)
        for name, loader in loaders.items()
    }
