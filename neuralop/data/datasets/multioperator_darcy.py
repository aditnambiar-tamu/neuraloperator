from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import DataLoader, Dataset

from ..transforms.data_processors import DataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer


class MultiOperatorDarcyDataProcessor(DataProcessor):
    """Data processor for episodic multi-operator Darcy batches."""

    def __init__(self, in_normalizer=None, out_normalizer=None):
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                data_dict[key] = value.to(self.device)

        if self.in_normalizer is not None:
            data_dict["u_context"] = self.in_normalizer.transform(data_dict["u_context"])
            data_dict["u_query"] = self.in_normalizer.transform(data_dict["u_query"])

        if self.out_normalizer is not None:
            data_dict["f_context"] = self.out_normalizer.transform(data_dict["f_context"])
            if self.training:
                data_dict["y"] = self.out_normalizer.transform(data_dict["y"])

        return data_dict

    def postprocess(self, output, data_dict):
        if self.out_normalizer is not None and not self.training:
            output = self.out_normalizer.inverse_transform(output)
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict


def _validate_multioperator_tensors(k, f, u, source="dataset"):
    """Validate scalar- or tensor-coefficient Darcy data shapes.

    The forcing and solution determine the spatial dimensionality. Coefficients
    may be scalar fields or channel-last square tensor fields.
    """
    if not all(torch.is_tensor(value) for value in (k, f, u)):
        raise TypeError(f"Expected k, f, and u in {source} to be tensors.")
    if f.ndim not in (3, 4) or u.ndim not in (3, 4):
        raise ValueError(
            "Expected f and u to contain 1D or 2D fields with shape "
            f"(n_operators, n_pairs, *spatial_shape), got {f.shape} and {u.shape}."
        )
    if f.shape != u.shape:
        raise ValueError(
            f"Expected f and u in {source} to have matching shapes, "
            f"got {f.shape} and {u.shape}."
        )

    spatial_shape = tuple(f.shape[2:])
    spatial_ndim = len(spatial_shape)
    coefficient_prefix = (f.shape[0], *spatial_shape)
    if (
        k.ndim < spatial_ndim + 1
        or tuple(k.shape[: spatial_ndim + 1]) != coefficient_prefix
    ):
        raise ValueError(
            f"Expected k in {source} to agree with f and u on operator count and "
            f"spatial shape {spatial_shape}, got k={k.shape}, f={f.shape}, u={u.shape}."
        )

    coefficient_shape = tuple(k.shape[spatial_ndim + 1 :])
    expected_tensor_shape = (spatial_ndim, spatial_ndim)
    if coefficient_shape not in ((), expected_tensor_shape):
        raise ValueError(
            f"Expected k in {source} to be a scalar field or a channel-last "
            f"{expected_tensor_shape} tensor field, got trailing shape "
            f"{coefficient_shape}."
        )
    if k.shape[0] == 0 or f.shape[1] == 0:
        raise ValueError(f"Expected {source} to contain at least one operator and pair.")

    return int(k.shape[0]), int(f.shape[1]), spatial_shape, coefficient_shape


class MultiOperatorDarcyDataset(Dataset):
    """Episodic dataset for in-context learning of Darcy operators.

    Expected raw tensor shapes are:

    - scalar ``k``: ``(n_operators, *spatial_shape)``
    - tensor ``k``: ``(n_operators, *spatial_shape, n_dim, n_dim)``
    - ``f``: ``(n_operators, n_pairs, *spatial_shape)``
    - ``u``: ``(n_operators, n_pairs, *spatial_shape)``

    Each item samples one operator and returns ``n_context`` input-output
    examples plus one query pair, in the format expected by ``FNOSets``.
    """

    def __init__(
        self,
        k: torch.Tensor,
        f: torch.Tensor,
        u: torch.Tensor,
        operator_indices: Union[List[int], torch.Tensor],
        n_context: int,
        n_samples: int = None,
        random_context: bool = True,
        include_k: bool = False,
        operator_direction: str = "f_to_u",
    ):
        super().__init__()

        _validate_multioperator_tensors(k, f, u)
        if f.shape[1] < n_context + 1:
            raise ValueError(
                f"Need at least n_context + 1 pairs per operator, got n_context={n_context} "
                f"and n_pairs={f.shape[1]}."
            )

        self.k = k.float()
        self.f = f.float()
        self.u = u.float()
        self.operator_indices = torch.as_tensor(operator_indices, dtype=torch.long)
        self.n_context = n_context
        self.n_pairs = f.shape[1]
        self.n_samples = n_samples if n_samples is not None else len(self.operator_indices)
        self.random_context = random_context
        self.include_k = include_k
        self.operator_direction = operator_direction

        if self.operator_direction not in {"f_to_u", "u_to_f"}:
            raise ValueError(
                "Expected operator_direction to be one of {'f_to_u', 'u_to_f'}, "
                f"got {self.operator_direction}."
            )

        if len(self.operator_indices) == 0:
            raise ValueError("operator_indices must contain at least one operator.")

    def __len__(self):
        return self.n_samples

    def _sample_pair_indices(self, index):
        if self.random_context:
            perm = torch.randperm(self.n_pairs)
            query_idx = perm[0]
            context_indices = perm[1 : self.n_context + 1]
        else:
            query_idx = (index // len(self.operator_indices)) % self.n_pairs
            available = torch.cat(
                [
                    torch.arange(0, query_idx, dtype=torch.long),
                    torch.arange(query_idx + 1, self.n_pairs, dtype=torch.long),
                ]
            )
            context_indices = available[: self.n_context]

        return context_indices, query_idx

    def __getitem__(self, index):
        operator_idx = self.operator_indices[index % len(self.operator_indices)]
        context_indices, query_idx = self._sample_pair_indices(index)

        if self.operator_direction == "f_to_u":
            input_functions = self.f
            output_functions = self.u
        else:
            input_functions = self.u
            output_functions = self.f

        sample = {
            "u_context": input_functions[operator_idx, context_indices].unsqueeze(1),
            "f_context": output_functions[operator_idx, context_indices].unsqueeze(1),
            "u_query": input_functions[operator_idx, query_idx].unsqueeze(0),
            "y": output_functions[operator_idx, query_idx].unsqueeze(0),
        }

        if self.include_k:
            sample["k"] = self.k[operator_idx].unsqueeze(0)

        return sample


def _fit_normalizer(data, encode, channel_dim=1):
    if not encode:
        return None

    reduce_dims = list(range(data.ndim))
    reduce_dims.pop(channel_dim)
    normalizer = UnitGaussianNormalizer(dim=reduce_dims)
    normalizer.fit(data)
    return normalizer


def load_multiop_darcy(
    data_path: Union[Path, str],
    n_train_operators: int,
    n_test_operators: int,
    n_context: int,
    batch_size: int,
    test_batch_size: int,
    n_train_samples: int = None,
    n_test_samples: int = None,
    encode_input: bool = True,
    encode_output: bool = True,
    include_k: bool = False,
    operator_direction: str = "f_to_u",
    num_workers: int = 0,
):
    """Load an episodic Darcy dataset for ``FNOSets`` training.

    The ``.pt`` file at ``data_path`` must contain keys ``k``, ``f``, and ``u``.
    Operators are split contiguously: the first ``n_train_operators`` are used
    for training, and the next ``n_test_operators`` are used for testing.
    """
    data_path = Path(data_path)
    data = torch.load(data_path.as_posix(), weights_only=False)

    required_keys = {"k", "f", "u"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise KeyError(f"Expected {data_path} to contain keys {required_keys}, missing {missing_keys}.")

    k = data["k"]
    f = data["f"]
    u = data["u"]

    if operator_direction not in {"f_to_u", "u_to_f"}:
        raise ValueError(
            "Expected operator_direction to be one of {'f_to_u', 'u_to_f'}, "
            f"got {operator_direction}."
        )

    n_operators = k.shape[0]
    if n_train_operators + n_test_operators > n_operators:
        raise ValueError(
            "Requested more train/test operators than available: "
            f"{n_train_operators} + {n_test_operators} > {n_operators}."
        )

    train_operator_indices = torch.arange(n_train_operators)
    test_operator_indices = torch.arange(n_train_operators, n_train_operators + n_test_operators)

    train_db = MultiOperatorDarcyDataset(
        k=k,
        f=f,
        u=u,
        operator_indices=train_operator_indices,
        n_context=n_context,
        n_samples=n_train_samples,
        random_context=True,
        include_k=include_k,
        operator_direction=operator_direction,
    )

    if n_test_samples is None:
        n_test_samples = n_test_operators * f.shape[1]
    test_db = MultiOperatorDarcyDataset(
        k=k,
        f=f,
        u=u,
        operator_indices=test_operator_indices,
        n_context=n_context,
        n_samples=n_test_samples,
        random_context=False,
        include_k=include_k,
        operator_direction=operator_direction,
    )

    train_u = u[train_operator_indices].reshape(-1, 1, *u.shape[2:]).float()
    train_f = f[train_operator_indices].reshape(-1, 1, *f.shape[2:]).float()

    if operator_direction == "f_to_u":
        train_inputs = train_f
        train_outputs = train_u
    else:
        train_inputs = train_u
        train_outputs = train_f

    data_processor = MultiOperatorDarcyDataProcessor(
        in_normalizer=_fit_normalizer(train_inputs, encode_input),
        out_normalizer=_fit_normalizer(train_outputs, encode_output),
    )

    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loaders = {
        "test": DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    }

    return train_loader, test_loaders, data_processor
