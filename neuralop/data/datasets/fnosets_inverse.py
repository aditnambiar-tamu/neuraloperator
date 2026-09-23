"""Data utilities for training :class:`FNOSetsInverse` on Darcy data.

The inverse processor consumes the forward episodic loaders directly and
replaces their query target with the operator coefficient. Scalar coefficient
fields remain one-channel targets. Symmetric 2D tensor coefficients are stored
channel-last in the datasets and represented to the model by the three
channel-first fields ``(K11, K22, K12)``. Reusing ``K12`` during reconstruction
guarantees symmetry, but direct component prediction does not guarantee
positive definiteness.
"""

from math import prod
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

SCALAR_COEFFICIENT_REPRESENTATION = "scalar"
SYMMETRIC_2D_COEFFICIENT_REPRESENTATION = "symmetric_2d"
SYMMETRIC_2D_COMPONENTS = ("K11", "K22", "K12")
_COEFFICIENT_CHANNELS = {
    SCALAR_COEFFICIENT_REPRESENTATION: 1,
    SYMMETRIC_2D_COEFFICIENT_REPRESENTATION: 3,
}


def symmetric_tensor_to_channels(
    coefficient,
    *,
    check_symmetry=True,
    symmetry_atol=1e-6,
    symmetry_rtol=1e-5,
):
    """Convert a symmetric 2D tensor field to ``(K11, K22, K12)`` channels.

    ``coefficient`` must have shape ``(..., height, width, 2, 2)``. It must
    not contain the singleton coefficient channel added by the Darcy datasets;
    the inverse processor removes that dimension first. The returned shape is
    ``(..., 3, height, width)``.
    """

    if not torch.is_tensor(coefficient):
        raise TypeError("coefficient must be a torch.Tensor.")
    if coefficient.ndim < 4 or tuple(coefficient.shape[-2:]) != (2, 2):
        raise ValueError(
            "Expected a 2D tensor coefficient with shape "
            "(..., height, width, 2, 2), got "
            f"{tuple(coefficient.shape)}."
        )

    k12_raw = coefficient[..., 0, 1]
    k21 = coefficient[..., 1, 0]
    if check_symmetry and not torch.allclose(
        k12_raw, k21, atol=symmetry_atol, rtol=symmetry_rtol
    ):
        max_error = (k12_raw - k21).abs().max().item()
        raise ValueError(
            "Expected a symmetric coefficient tensor with K12 == K21 "
            f"within atol={symmetry_atol} and rtol={symmetry_rtol}; maximum "
            f"absolute difference was {max_error:.6g}."
        )

    # Averaging removes harmless roundoff asymmetry after it has been checked.
    k12 = 0.5 * (k12_raw + k21)
    return torch.stack(
        (coefficient[..., 0, 0], coefficient[..., 1, 1], k12),
        dim=-3,
    )


def channels_to_symmetric_tensor(coefficient):
    """Reconstruct ``(..., H, W, 2, 2)`` matrices from three channels."""

    if not torch.is_tensor(coefficient):
        raise TypeError("coefficient must be a torch.Tensor.")
    if coefficient.ndim < 3 or coefficient.shape[-3] != 3:
        raise ValueError(
            "Expected symmetric coefficient channels with shape "
            "(..., 3, height, width), got "
            f"{tuple(coefficient.shape)}."
        )

    k11 = coefficient.select(-3, 0)
    k22 = coefficient.select(-3, 1)
    k12 = coefficient.select(-3, 2)
    first_row = torch.stack((k11, k12), dim=-1)
    second_row = torch.stack((k12, k22), dim=-1)
    return torch.stack((first_row, second_row), dim=-2)


def relative_frobenius_error(y_pred, y, eps=1e-8, reduction="mean"):
    """Compute relative matrix-field error for symmetric tensor channels.

    The error is computed independently for every batch element. Because the
    represented matrix is symmetric, the squared component weights are
    ``(1, 1, 2)`` for ``(K11, K22, K12)``.
    """

    if y_pred.shape != y.shape:
        raise ValueError(
            "Prediction and target must have identical shapes, got "
            f"{tuple(y_pred.shape)} and {tuple(y.shape)}."
        )
    if y_pred.ndim < 4 or y_pred.shape[1] != 3:
        raise ValueError(
            "Expected batched symmetric coefficient channels with shape "
            "(batch, 3, *spatial_shape), got "
            f"{tuple(y_pred.shape)}."
        )
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(
            "reduction must be one of 'none', 'mean', or 'sum', got "
            f"{reduction!r}."
        )

    weight_shape = (1, 3) + (1,) * (y_pred.ndim - 2)
    weights = y_pred.new_tensor((1.0, 1.0, 2.0)).reshape(weight_shape)
    reduce_dims = tuple(range(1, y_pred.ndim))
    numerator = ((y_pred - y).square() * weights).sum(dim=reduce_dims).sqrt()
    denominator = (y.square() * weights).sum(dim=reduce_dims).sqrt()
    error = numerator / (denominator + eps)

    if reduction == "sum":
        return error.sum()
    if reduction == "mean":
        return error.mean()
    return error


class RelativeFrobeniusLoss:
    """Trainer-compatible relative Frobenius metric for symmetric tensors."""

    def __init__(self, eps=1e-8, reduction="sum"):
        if reduction not in {"mean", "sum"}:
            raise ValueError("RelativeFrobeniusLoss reduction must be 'mean' or 'sum'.")
        self.eps = eps
        self.reduction = reduction

    @property
    def name(self):
        return "relative_frobenius"

    def __call__(self, y_pred, y, **kwargs):
        return relative_frobenius_error(
            y_pred,
            y,
            eps=self.eps,
            reduction=self.reduction,
        )


def smallest_symmetric_eigenvalue(coefficient):
    """Return the smallest eigenvalue at every point of a 3-channel field."""

    if coefficient.ndim < 3 or coefficient.shape[-3] != 3:
        raise ValueError(
            "Expected symmetric coefficient channels with shape "
            "(..., 3, height, width), got "
            f"{tuple(coefficient.shape)}."
        )
    k11 = coefficient.select(-3, 0)
    k22 = coefficient.select(-3, 1)
    k12 = coefficient.select(-3, 2)
    discriminant = ((k11 - k22).square() + 4 * k12.square()).clamp_min(0)
    return 0.5 * (k11 + k22 - discriminant.sqrt())


def _validate_coefficient_representation(coefficient_representation):
    if coefficient_representation not in _COEFFICIENT_CHANNELS:
        raise ValueError(
            "coefficient_representation must be one of "
            f"{tuple(_COEFFICIENT_CHANNELS)}, got "
            f"{coefficient_representation!r}."
        )
    return coefficient_representation


def _dataset_coefficient_spec(dataset):
    if isinstance(dataset, ChunkedMultiOperatorDarcyDataset):
        coefficient_shape = tuple(dataset.coefficient_shape)
        spatial_ndim = len(dataset.spatial_shape)
    elif isinstance(dataset, MultiOperatorDarcyDataset):
        spatial_ndim = dataset.f.ndim - 2
        coefficient_shape = tuple(dataset.k.shape[spatial_ndim + 1 :])
    else:
        raise TypeError(
            "Expected a MultiOperatorDarcyDataset or "
            "ChunkedMultiOperatorDarcyDataset."
        )

    if coefficient_shape == ():
        inferred_representation = SCALAR_COEFFICIENT_REPRESENTATION
    elif spatial_ndim == 2 and coefficient_shape == (2, 2):
        inferred_representation = SYMMETRIC_2D_COEFFICIENT_REPRESENTATION
    else:
        raise ValueError(
            "FNOSetsInverse supports scalar coefficient fields and symmetric "
            "2D tensor fields with trailing shape (2, 2); got spatial_ndim="
            f"{spatial_ndim} and coefficient_shape={coefficient_shape}."
        )
    return spatial_ndim, coefficient_shape, inferred_representation


def _resolve_coefficient_representation(dataset, coefficient_representation=None):
    spatial_ndim, coefficient_shape, inferred = _dataset_coefficient_spec(dataset)
    if coefficient_representation is None:
        coefficient_representation = inferred
    coefficient_representation = _validate_coefficient_representation(
        coefficient_representation
    )
    if coefficient_representation != inferred:
        raise ValueError(
            f"Dataset coefficient shape {coefficient_shape} implies "
            f"coefficient_representation={inferred!r}, but received "
            f"{coefficient_representation!r}."
        )
    return spatial_ndim, coefficient_representation


def _stored_coefficients_to_channels(
    coefficient,
    coefficient_representation,
    spatial_ndim,
    *,
    symmetry_atol,
    symmetry_rtol,
):
    """Convert stored ``(operator, *spatial, *matrix)`` data to channels."""

    if coefficient_representation == SCALAR_COEFFICIENT_REPRESENTATION:
        expected_ndim = spatial_ndim + 1
        if coefficient.ndim != expected_ndim:
            raise ValueError(
                f"Expected stored scalar coefficients with {expected_ndim} "
                f"dimensions, got shape {tuple(coefficient.shape)}."
            )
        return coefficient.unsqueeze(1)

    expected_ndim = spatial_ndim + 3
    if coefficient.ndim != expected_ndim:
        raise ValueError(
            f"Expected stored tensor coefficients with {expected_ndim} "
            f"dimensions, got shape {tuple(coefficient.shape)}."
        )
    return symmetric_tensor_to_channels(
        coefficient,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
    )


def _validate_coefficient_normalizer(
    coefficient_normalizer,
    coefficient_representation,
    spatial_ndim,
):
    if coefficient_normalizer is None:
        return
    mean = coefficient_normalizer.mean
    std = coefficient_normalizer.std
    if mean is None or std is None:
        raise ValueError("coefficient_normalizer must be fitted before use.")
    if mean.shape != std.shape or mean.ndim < 2:
        raise ValueError(
            "coefficient_normalizer mean and std must have the same "
            f"channel-first shape, got {tuple(mean.shape)} and "
            f"{tuple(std.shape)}."
        )

    expected_channels = _COEFFICIENT_CHANNELS[coefficient_representation]
    valid_shape = mean.shape[0] == 1 and mean.shape[1] == expected_channels
    valid_shape = valid_shape and all(size == 1 for size in mean.shape[2:])
    if spatial_ndim is not None:
        valid_shape = valid_shape and mean.ndim == spatial_ndim + 2
    if not valid_shape:
        expected_spatial = (
            "one singleton per spatial dimension"
            if spatial_ndim is None
            else f"{spatial_ndim} trailing singleton dimensions"
        )
        raise ValueError(
            "coefficient_normalizer has an incompatible shape: expected "
            f"(1, {expected_channels}, {expected_spatial}), got "
            f"{tuple(mean.shape)}."
        )


class CoefficientTargetDataset(Dataset):
    """Use an episodic dataset's coefficient field as its target."""

    def __init__(
        self,
        dataset,
        coefficient_representation=None,
        symmetry_atol=1e-6,
        symmetry_rtol=1e-5,
    ):
        if not getattr(dataset, "include_k", False):
            raise ValueError(
                "CoefficientTargetDataset requires a dataset created with "
                "include_k=True."
            )
        spatial_ndim, coefficient_representation = _resolve_coefficient_representation(
            dataset,
            coefficient_representation=coefficient_representation,
        )
        self.dataset = dataset
        self.spatial_ndim = spatial_ndim
        self.coefficient_representation = coefficient_representation
        self.coefficient_channels = _COEFFICIENT_CHANNELS[coefficient_representation]
        self.symmetry_atol = symmetry_atol
        self.symmetry_rtol = symmetry_rtol

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        if self.coefficient_representation == SCALAR_COEFFICIENT_REPRESENTATION:
            sample["y"] = sample["k"]
        else:
            coefficient = sample["k"]
            if coefficient.shape[0] != 1:
                raise ValueError(
                    "Expected the dataset coefficient target to have a leading "
                    f"singleton channel, got shape {tuple(coefficient.shape)}."
                )
            sample["y"] = symmetric_tensor_to_channels(
                coefficient.squeeze(0),
                symmetry_atol=self.symmetry_atol,
                symmetry_rtol=self.symmetry_rtol,
            )
        return sample


class FNOSetsInverseDataProcessor(DataProcessor):
    """Normalize context functions and scalar or symmetric tensor targets."""

    def __init__(
        self,
        input_normalizer=None,
        output_normalizer=None,
        coefficient_normalizer=None,
        coefficient_representation=SCALAR_COEFFICIENT_REPRESENTATION,
        spatial_ndim=None,
        symmetry_atol=1e-6,
        symmetry_rtol=1e-5,
    ):
        super().__init__()
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.coefficient_normalizer = coefficient_normalizer
        self.coefficient_representation = _validate_coefficient_representation(
            coefficient_representation
        )
        if (
            self.coefficient_representation == SYMMETRIC_2D_COEFFICIENT_REPRESENTATION
            and spatial_ndim not in (None, 2)
        ):
            raise ValueError("The symmetric_2d representation requires spatial_ndim=2.")
        self.spatial_ndim = spatial_ndim
        self.symmetry_atol = symmetry_atol
        self.symmetry_rtol = symmetry_rtol
        _validate_coefficient_normalizer(
            coefficient_normalizer,
            self.coefficient_representation,
            spatial_ndim,
        )
        self.device = "cpu"
        self.model = None

    @property
    def coefficient_channels(self):
        """Number of output channels required from ``FNOSetsInverse``."""

        return _COEFFICIENT_CHANNELS[self.coefficient_representation]

    @property
    def coefficient_components(self):
        if self.coefficient_representation == SYMMETRIC_2D_COEFFICIENT_REPRESENTATION:
            return SYMMETRIC_2D_COMPONENTS
        return ("k",)

    def to(self, device):
        if self.input_normalizer is not None:
            self.input_normalizer = self.input_normalizer.to(device)
        if self.output_normalizer is not None:
            self.output_normalizer = self.output_normalizer.to(device)
        if self.coefficient_normalizer is not None:
            self.coefficient_normalizer = self.coefficient_normalizer.to(device)
        self.device = device
        return self

    def encode_coefficient(self, coefficient, batched=True):
        """Convert a raw loader coefficient to the model target layout."""

        if self.coefficient_representation == SCALAR_COEFFICIENT_REPRESENTATION:
            return coefficient

        coefficient_channel_dim = 1 if batched else 0
        expected_ndim = 6 if batched else 5
        if (
            coefficient.ndim != expected_ndim
            or coefficient.shape[coefficient_channel_dim] != 1
        ):
            if batched:
                expected = "(batch, 1, height, width, 2, 2)"
            else:
                expected = "(1, height, width, 2, 2)"
            raise ValueError(
                f"Expected raw tensor coefficient shape {expected}, got "
                f"{tuple(coefficient.shape)}."
            )
        coefficient = coefficient.squeeze(coefficient_channel_dim)
        return symmetric_tensor_to_channels(
            coefficient,
            symmetry_atol=self.symmetry_atol,
            symmetry_rtol=self.symmetry_rtol,
        )

    def decode_coefficient(self, coefficient):
        """Convert model coefficient channels back to physical matrices."""

        if self.coefficient_representation == SCALAR_COEFFICIENT_REPRESENTATION:
            return coefficient
        return channels_to_symmetric_tensor(coefficient)

    def preprocess(self, data_dict, batched=True):
        if "k" in data_dict:
            # Encode and validate tensor coefficients while ordinary DataLoader
            # batches are still on CPU, avoiding a GPU synchronization from the
            # symmetry check.
            data_dict["y"] = self.encode_coefficient(
                data_dict["k"], batched=batched
            )
        elif self.training:
            raise KeyError(
                "FNOSetsInverseDataProcessor requires training batches "
                "containing k. Create the dataset with include_k=True."
            )
        else:
            # A forward-loader y is not an inverse coefficient target. During
            # context-only inference, omit it so the model makes an unlabeled
            # coefficient prediction.
            data_dict.pop("y", None)

        for key, value in data_dict.items():
            if torch.is_tensor(value):
                data_dict[key] = value.to(self.device)

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
        if output.ndim < 2 or output.shape[1] != self.coefficient_channels:
            output_channels = output.shape[1] if output.ndim >= 2 else None
            raise ValueError(
                "FNOSetsInverse output channels do not match the coefficient "
                f"representation: expected {self.coefficient_channels}, got "
                f"{output_channels}."
            )
        if "y" in data_dict and output.shape != data_dict["y"].shape:
            raise ValueError(
                "FNOSetsInverse output and coefficient target shapes must match; "
                f"got output={tuple(output.shape)} and target="
                f"{tuple(data_dict['y'].shape)}. Construct FNOSetsInverse with "
                f"coefficient_channels={self.coefficient_channels}."
            )
        if self.coefficient_normalizer is not None and not self.training:
            output = self.coefficient_normalizer.inverse_transform(output)
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict


def _update_channel_stats(values, count, mean, m2):
    values = values.to(dtype=torch.float64)
    reduce_dims = (0,) + tuple(range(2, values.ndim))
    batch_count = prod(values.shape[dim] for dim in reduce_dims)
    batch_mean = values.mean(dim=reduce_dims)
    mean_shape = (1, -1) + (1,) * (values.ndim - 2)
    batch_m2 = (values - batch_mean.reshape(mean_shape)).square().sum(dim=reduce_dims)

    if count == 0:
        return batch_count, batch_mean, batch_m2

    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * batch_count / total
    m2 = m2 + batch_m2 + delta.square() * count * batch_count / total
    return total, mean, m2


def fit_coefficient_normalizer(
    dataset,
    coefficient_representation=None,
    *,
    symmetry_atol=1e-6,
    symmetry_rtol=1e-5,
):
    """Fit per-channel coefficient statistics using training operators only.

    Chunked datasets are processed one file at a time, so fitting does not
    materialize the complete coefficient dataset in memory. Every training
    operator contributes once, independently of ``n_samples``.
    """

    if isinstance(dataset, CoefficientTargetDataset):
        if coefficient_representation is None:
            coefficient_representation = dataset.coefficient_representation
        symmetry_atol = dataset.symmetry_atol
        symmetry_rtol = dataset.symmetry_rtol
        dataset = dataset.dataset

    spatial_ndim, coefficient_representation = _resolve_coefficient_representation(
        dataset,
        coefficient_representation=coefficient_representation,
    )
    count, mean, m2 = 0, None, None

    def update(coefficient, count, mean, m2):
        coefficient = _stored_coefficients_to_channels(
            coefficient,
            coefficient_representation,
            spatial_ndim,
            symmetry_atol=symmetry_atol,
            symmetry_rtol=symmetry_rtol,
        )
        return _update_channel_stats(coefficient, count, mean, m2)

    if isinstance(dataset, ChunkedMultiOperatorDarcyDataset):
        for path in dataset.chunk_paths:
            chunk = _load_chunk(path)
            _validate_chunk(chunk, path)
            count, mean, m2 = update(chunk["k"], count, mean, m2)
            del chunk
    else:
        coefficient = dataset.k[dataset.operator_indices]
        count, mean, m2 = update(coefficient, count, mean, m2)

    if count < 2:
        raise ValueError(
            "At least two coefficient values per channel are required to "
            "normalize k."
        )

    coefficient_channels = _COEFFICIENT_CHANNELS[coefficient_representation]
    stat_shape = (1, coefficient_channels) + (1,) * spatial_ndim
    return UnitGaussianNormalizer(
        mean=mean.float().reshape(stat_shape),
        std=(m2 / (count - 1)).clamp_min(0).sqrt().float().reshape(stat_shape),
        dim=[0] + list(range(2, spatial_ndim + 2)),
    )


def wrap_fnosets_inverse_loader(
    loader,
    coefficient_representation=None,
    *,
    symmetry_atol=1e-6,
    symmetry_rtol=1e-5,
):
    """Return a coefficient-target loader while preserving its batch sampler."""

    if isinstance(loader.dataset, CoefficientTargetDataset):
        if (
            coefficient_representation is not None
            and coefficient_representation
            != loader.dataset.coefficient_representation
        ):
            raise ValueError(
                "Loader is already wrapped with coefficient_representation="
                f"{loader.dataset.coefficient_representation!r}, which does not "
                f"match {coefficient_representation!r}."
            )
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

    target_dataset = CoefficientTargetDataset(
        loader.dataset,
        coefficient_representation=coefficient_representation,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
    )
    return DataLoader(target_dataset, **kwargs)


def build_fnosets_inverse_data_processor(
    base_data_processor,
    training_dataset,
    coefficient_normalizer: Optional[UnitGaussianNormalizer] = None,
    coefficient_representation=None,
    *,
    symmetry_atol=1e-6,
    symmetry_rtol=1e-5,
):
    """Build an inverse processor from a forward loader's normalizers.

    The representation is inferred from the dataset when omitted. Scalar
    fields use ``"scalar"`` and 2D matrix fields use ``"symmetric_2d"``.
    """

    if isinstance(training_dataset, CoefficientTargetDataset):
        if coefficient_representation is None:
            coefficient_representation = training_dataset.coefficient_representation
        symmetry_atol = training_dataset.symmetry_atol
        symmetry_rtol = training_dataset.symmetry_rtol
        base_dataset = training_dataset.dataset
    else:
        base_dataset = training_dataset

    if not getattr(base_dataset, "include_k", False):
        raise ValueError(
            "FNOSetsInverse training requires a dataset created with "
            "include_k=True."
        )
    spatial_ndim, coefficient_representation = _resolve_coefficient_representation(
        base_dataset,
        coefficient_representation=coefficient_representation,
    )
    if coefficient_normalizer is None:
        coefficient_normalizer = fit_coefficient_normalizer(
            base_dataset,
            coefficient_representation=coefficient_representation,
            symmetry_atol=symmetry_atol,
            symmetry_rtol=symmetry_rtol,
        )
    else:
        _validate_coefficient_normalizer(
            coefficient_normalizer,
            coefficient_representation,
            spatial_ndim,
        )

    return FNOSetsInverseDataProcessor(
        input_normalizer=base_data_processor.in_normalizer,
        output_normalizer=base_data_processor.out_normalizer,
        coefficient_normalizer=coefficient_normalizer,
        coefficient_representation=coefficient_representation,
        spatial_ndim=spatial_ndim,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
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
    """Save inverse normalization and coefficient representation state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "input_normalizer": _normalizer_checkpoint(data_processor.input_normalizer),
            "output_normalizer": _normalizer_checkpoint(
                data_processor.output_normalizer
            ),
            "coefficient_normalizer": _normalizer_checkpoint(
                data_processor.coefficient_normalizer
            ),
            "coefficient_representation": data_processor.coefficient_representation,
            "spatial_ndim": data_processor.spatial_ndim,
            "symmetry_atol": data_processor.symmetry_atol,
            "symmetry_rtol": data_processor.symmetry_rtol,
            "metadata": {} if metadata is None else metadata,
        },
        path,
    )


def load_fnosets_inverse_data_processor(path, device="cpu"):
    """Load inverse normalization and representation state.

    Checkpoints written before tensor support are interpreted as scalar
    coefficient processors.
    """

    checkpoint = torch.load(
        Path(path).as_posix(), map_location="cpu", weights_only=False
    )
    format_version = checkpoint.get("format_version", 1)
    if format_version not in {1, 2}:
        raise ValueError(
            "Unsupported FNOSetsInverse data processor checkpoint format "
            f"version {format_version}."
        )
    data_processor = FNOSetsInverseDataProcessor(
        input_normalizer=_normalizer_from_checkpoint(checkpoint["input_normalizer"]),
        output_normalizer=_normalizer_from_checkpoint(checkpoint["output_normalizer"]),
        coefficient_normalizer=_normalizer_from_checkpoint(
            checkpoint["coefficient_normalizer"]
        ),
        coefficient_representation=checkpoint.get(
            "coefficient_representation", SCALAR_COEFFICIENT_REPRESENTATION
        ),
        spatial_ndim=checkpoint.get("spatial_ndim"),
        symmetry_atol=checkpoint.get("symmetry_atol", 1e-6),
        symmetry_rtol=checkpoint.get("symmetry_rtol", 1e-5),
    ).to(device)
    return data_processor, checkpoint.get("metadata", {})


def wrap_fnosets_inverse_loaders(
    loaders: Mapping[str, DataLoader],
    coefficient_representation=None,
    *,
    symmetry_atol=1e-6,
    symmetry_rtol=1e-5,
):
    """Wrap every loader in a named validation/test loader dictionary."""

    return {
        name: wrap_fnosets_inverse_loader(
            loader,
            coefficient_representation=coefficient_representation,
            symmetry_atol=symmetry_atol,
            symmetry_rtol=symmetry_rtol,
        )
        for name, loader in loaders.items()
    }
