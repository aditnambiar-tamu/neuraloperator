from pathlib import Path

import pytest
import torch

from neuralop.data.datasets import (
    ChunkedMultiOperatorDarcyDataset,
    MultiOperatorDarcyDataset,
    load_chunked_multiop_darcy,
    load_multiop_darcy,
)


def _write_chunk(path, value, spatial_shape=(5,), n_operators=2, n_pairs=4):
    shape = (n_operators, n_pairs, *spatial_shape)
    f = torch.full(shape, float(value))
    pair_offsets = torch.arange(n_pairs).reshape(
        1, n_pairs, *((1,) * len(spatial_shape))
    )
    f = f + pair_offsets
    torch.save(
        {
            "k": torch.full((n_operators, *spatial_shape), float(value + 1)),
            "f": f,
            "u": f + 100,
            "operator_start": value * n_operators,
        },
        path,
    )


def _write_tensor_coefficient_chunk(
    path, value, spatial_shape=(3, 4), n_operators=2, n_pairs=4
):
    shape = (n_operators, n_pairs, *spatial_shape)
    torch.save(
        {
            "k": torch.full(
                (n_operators, *spatial_shape, 2, 2), float(value + 1)
            ),
            "f": torch.full(shape, float(value)),
            "u": torch.full(shape, float(value + 100)),
        },
        path,
    )


@pytest.fixture
def chunk_dir(tmp_path):
    # Names are intentionally created out of order to test sorted discovery.
    for index in (3, 1, 4, 0, 2):
        _write_chunk(tmp_path / f"chunk_{index:02d}.pt", index * 10)
    torch.save({"n_chunks": 5}, tmp_path / "manifest.pt")
    return tmp_path


def _load(chunk_dir, **kwargs):
    defaults = {
        "chunk_dir": chunk_dir,
        "n_context": 2,
        "batch_size": 2,
        "test_batch_size": 2,
        "n_train_samples": 8,
        "n_val_samples": 4,
        "n_test_samples": 4,
        "n_train_chunks": 2,
        "n_val_chunks": 1,
        "n_test_chunks": 1,
    }
    defaults.update(kwargs)
    return load_chunked_multiop_darcy(**defaults)


def test_count_splits_are_sorted_disjoint_and_val_is_trainer_ready(chunk_dir):
    train_loader, val_loaders, test_loader, _ = _load(
        chunk_dir, encode_input=False, encode_output=False
    )

    train_names = [path.name for path in train_loader.dataset.chunk_paths]
    val_names = [path.name for path in val_loaders["val"].dataset.chunk_paths]
    test_names = [path.name for path in test_loader.dataset.chunk_paths]

    assert train_names == ["chunk_00.pt", "chunk_01.pt"]
    assert val_names == ["chunk_02.pt"]
    assert test_names == ["chunk_03.pt"]
    assert not (set(train_names) & set(val_names))
    assert not (set(train_names) & set(test_names))
    assert not (set(val_names) & set(test_names))
    assert set(val_loaders) == {"val"}


def test_discovery_uses_natural_order_and_ignores_unrelated_pt_files(tmp_path):
    for index in (10, 2, 1):
        _write_chunk(tmp_path / f"chunk_{index}.pt", index)
    torch.save({"model": torch.ones(1)}, tmp_path / "checkpoint.pt")
    torch.save({"n_chunks": 3}, tmp_path / "manifest.pt")

    train_loader, val_loaders, test_loader, _ = load_chunked_multiop_darcy(
        chunk_dir=tmp_path,
        n_context=2,
        batch_size=1,
        n_train_chunks=1,
        n_val_chunks=1,
        n_test_chunks=1,
        encode_input=False,
        encode_output=False,
    )

    assert train_loader.dataset.chunk_paths[0].name == "chunk_1.pt"
    assert val_loaders["val"].dataset.chunk_paths[0].name == "chunk_2.pt"
    assert test_loader.dataset.chunk_paths[0].name == "chunk_10.pt"


def test_shapes_direction_and_include_k_for_1d_and_2d(tmp_path):
    for index in range(3):
        _write_chunk(
            tmp_path / f"chunk_field_{index}.pt",
            index,
            spatial_shape=(3, 4),
        )

    train_loader, _, _, _ = load_chunked_multiop_darcy(
        chunk_dir=tmp_path,
        n_context=2,
        batch_size=2,
        n_train_chunks=1,
        n_val_chunks=1,
        n_test_chunks=1,
        n_train_samples=2,
        n_val_samples=2,
        n_test_samples=2,
        encode_input=False,
        encode_output=False,
        include_k=True,
        operator_direction="u_to_f",
    )
    sample = next(iter(train_loader))
    assert sample["u_context"].shape == (2, 2, 1, 3, 4)
    assert sample["f_context"].shape == (2, 2, 1, 3, 4)
    assert sample["u_query"].shape == (2, 1, 3, 4)
    assert sample["y"].shape == (2, 1, 3, 4)
    assert sample["k"].shape == (2, 1, 3, 4)
    assert torch.all(sample["u_context"] >= 100)
    assert torch.all(sample["f_context"] < 100)


def test_tensor_coefficients_preserve_forward_sample_shapes(tmp_path):
    for index in range(3):
        _write_tensor_coefficient_chunk(tmp_path / f"chunk_{index}.pt", index)

    train_loader, _, _, _ = load_chunked_multiop_darcy(
        chunk_dir=tmp_path,
        n_context=2,
        batch_size=2,
        n_train_chunks=1,
        n_val_chunks=1,
        n_test_chunks=1,
        n_train_samples=2,
        n_val_samples=2,
        n_test_samples=2,
        encode_input=False,
        encode_output=False,
        include_k=True,
    )
    sample = next(iter(train_loader))

    assert train_loader.dataset.spatial_shape == (3, 4)
    assert train_loader.dataset.coefficient_shape == (2, 2)
    assert sample["u_context"].shape == (2, 2, 1, 3, 4)
    assert sample["f_context"].shape == (2, 2, 1, 3, 4)
    assert sample["u_query"].shape == (2, 1, 3, 4)
    assert sample["y"].shape == (2, 1, 3, 4)
    assert sample["k"].shape == (2, 1, 3, 4, 2, 2)


def test_in_memory_dataset_accepts_tensor_coefficients():
    k = torch.ones(3, 3, 4, 2, 2)
    f = torch.ones(3, 4, 3, 4)
    dataset = MultiOperatorDarcyDataset(
        k=k,
        f=f,
        u=f + 1,
        operator_indices=torch.arange(3),
        n_context=2,
        random_context=False,
        include_k=True,
    )

    sample = dataset[0]
    assert sample["u_query"].shape == (1, 3, 4)
    assert sample["k"].shape == (1, 3, 4, 2, 2)


def test_chunks_cannot_mix_scalar_and_tensor_coefficients(tmp_path):
    _write_chunk(tmp_path / "chunk_0.pt", 0, spatial_shape=(3, 4))
    _write_tensor_coefficient_chunk(tmp_path / "chunk_1.pt", 1)

    with pytest.raises(ValueError, match="incompatible with earlier chunks"):
        ChunkedMultiOperatorDarcyDataset(
            [tmp_path / "chunk_0.pt", tmp_path / "chunk_1.pt"],
            n_context=2,
        )


def test_in_memory_dataset_uses_seven_fixed_context_pairs():
    pair_values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    f = pair_values.expand(2, 8, 5).clone()
    dataset = MultiOperatorDarcyDataset(
        k=torch.ones(2, 5),
        f=f,
        u=f + 100,
        operator_indices=torch.arange(2),
        n_context=7,
        n_samples=8,
        random_context=True,
        fixed_context_indices=range(7),
    )

    for index in range(len(dataset)):
        sample = dataset[index]
        assert torch.equal(
            sample["u_context"][:, 0, 0],
            torch.arange(7, dtype=torch.float32),
        )
        assert torch.all(sample["u_query"] == 7)


def test_in_memory_loader_propagates_fixed_context_indices(tmp_path):
    pair_values = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    f = pair_values.expand(4, 5, 3).clone()
    data_path = tmp_path / "dataset.pt"
    torch.save({"k": torch.ones(4, 3), "f": f, "u": f + 100}, data_path)

    train_loader, test_loaders, _ = load_multiop_darcy(
        data_path=data_path,
        n_train_operators=2,
        n_test_operators=2,
        n_context=2,
        batch_size=2,
        test_batch_size=2,
        n_train_samples=2,
        encode_input=False,
        encode_output=False,
        fixed_context_indices=[1, 3],
    )

    train_sample = next(iter(train_loader))
    assert torch.equal(
        train_sample["u_context"][:, :, 0, 0],
        torch.tensor([[1.0, 3.0], [1.0, 3.0]]),
    )
    assert len(test_loaders["test"].dataset) == 2 * 3


def test_chunked_loader_cycles_queries_outside_fixed_context(tmp_path):
    for index in range(3):
        _write_chunk(
            tmp_path / f"chunk_{index}.pt",
            index * 10,
            n_operators=2,
            n_pairs=5,
        )

    train_loader, val_loaders, test_loader, _ = load_chunked_multiop_darcy(
        chunk_dir=tmp_path,
        n_context=2,
        batch_size=2,
        n_train_chunks=1,
        n_val_chunks=1,
        n_test_chunks=1,
        n_train_samples=2,
        encode_input=False,
        encode_output=False,
        fixed_context_indices=[1, 3],
    )

    assert torch.equal(
        train_loader.dataset.fixed_context_indices, torch.tensor([1, 3])
    )
    assert len(val_loaders["val"].dataset) == 2 * 3
    assert len(test_loader.dataset) == 2 * 3

    val_dataset = val_loaders["val"].dataset
    samples = [val_dataset[(0, episode)] for episode in (0, 2, 4)]
    assert [sample["u_query"][0, 0].item() for sample in samples] == [
        10,
        12,
        14,
    ]
    for sample in samples:
        assert torch.equal(
            sample["u_context"][:, 0, 0], torch.tensor([11.0, 13.0])
        )


@pytest.mark.parametrize(
    ("fixed_context_indices", "message"),
    [
        ([0], "exactly n_context"),
        ([0, 0], "must not contain duplicates"),
        ([0, 4], "must lie in"),
    ],
)
def test_fixed_context_indices_are_validated(fixed_context_indices, message):
    f = torch.ones(2, 4, 5)
    with pytest.raises(ValueError, match=message):
        MultiOperatorDarcyDataset(
            k=torch.ones(2, 5),
            f=f,
            u=f,
            operator_indices=torch.arange(2),
            n_context=2,
            fixed_context_indices=fixed_context_indices,
        )


def test_validation_is_deterministic(chunk_dir):
    _, val_loaders, _, _ = _load(
        chunk_dir, encode_input=False, encode_output=False
    )
    val_loader = val_loaders["val"]
    first = [
        {key: value.clone() for key, value in batch.items()}
        for batch in val_loader
    ]
    val_loader.dataset.clear_cache()
    second = list(val_loader)

    assert len(first) == len(second)
    for first_batch, second_batch in zip(first, second):
        for key in first_batch:
            assert torch.equal(first_batch[key], second_batch[key])


def test_chunk_aware_sampler_loads_each_training_chunk_once(chunk_dir):
    train_loader, _, _, _ = _load(
        chunk_dir, encode_input=False, encode_output=False
    )
    dataset = train_loader.dataset
    dataset.clear_cache()
    dataset.chunk_load_count = 0

    list(train_loader)

    assert dataset.chunk_load_count == len(dataset.chunk_paths)


def test_training_samples_full_chunk_operator_range_when_samples_are_fewer(tmp_path):
    _write_chunk(tmp_path / "chunk_0.pt", 0, n_operators=20)
    _write_chunk(tmp_path / "chunk_1.pt", 10, n_operators=2)
    _write_chunk(tmp_path / "chunk_2.pt", 20, n_operators=2)
    train_loader, _, _, _ = load_chunked_multiop_darcy(
        chunk_dir=tmp_path,
        n_context=2,
        batch_size=4,
        n_train_chunks=1,
        n_val_chunks=1,
        n_test_chunks=1,
        n_train_samples=8,
        n_val_samples=2,
        n_test_samples=2,
        encode_input=False,
        encode_output=False,
        seed=3,
    )

    sampled_episodes = [
        episode
        for batch in train_loader.batch_sampler
        for _, episode in batch
    ]
    assert len(sampled_episodes) == 8
    assert max(sampled_episodes) >= 8


def test_uneven_chunks_and_deterministic_sample_cycles(tmp_path):
    _write_chunk(tmp_path / "chunk_0.pt", 0, n_operators=2)
    _write_chunk(tmp_path / "chunk_1.pt", 10, n_operators=3)
    dataset = ChunkedMultiOperatorDarcyDataset(
        [tmp_path / "chunk_0.pt", tmp_path / "chunk_1.pt"],
        n_context=2,
        n_samples=7,
        random_context=False,
    )

    assert len(dataset) == 7
    assert dataset.sample_counts == (4, 3)
    assert dataset._global_to_local(4) == (1, 2)
    assert dataset._global_to_local(5) == (0, 2)

    query_zero = dataset[(0, 0)]["u_query"]
    query_one = dataset[(0, 2)]["u_query"]
    assert torch.equal(query_one, query_zero + 1)


def test_normalizers_use_training_chunks_only(chunk_dir):
    _, _, _, processor = _load(chunk_dir)

    train_f = torch.cat(
        [
            torch.load(chunk_dir / "chunk_00.pt", weights_only=False)["f"],
            torch.load(chunk_dir / "chunk_01.pt", weights_only=False)["f"],
        ]
    ).reshape(-1, 1, 5)
    train_u = train_f + 100
    reduce_dims = [0, 2]

    assert torch.allclose(
        processor.in_normalizer.mean,
        train_f.mean(dim=reduce_dims, keepdim=True),
    )
    assert torch.allclose(
        processor.in_normalizer.std,
        train_f.std(dim=reduce_dims, keepdim=True),
    )
    assert torch.allclose(
        processor.out_normalizer.mean,
        train_u.mean(dim=reduce_dims, keepdim=True),
    )
    assert processor.in_normalizer.mean.item() < 20


def test_reverse_direction_swaps_normalizers(chunk_dir):
    _, _, _, processor = _load(chunk_dir, operator_direction="u_to_f")
    assert processor.in_normalizer.mean.item() > 100
    assert processor.out_normalizer.mean.item() < 20


def test_explicit_selections_and_overlap_validation(chunk_dir):
    names = [f"chunk_{index:02d}.pt" for index in range(5)]
    train_loader, val_loaders, test_loader, _ = load_chunked_multiop_darcy(
        chunk_dir=chunk_dir,
        n_context=2,
        batch_size=2,
        train_chunks=names[:2],
        val_chunks=[2],
        test_chunks=[Path(names[3])],
        n_train_samples=2,
        n_val_samples=2,
        n_test_samples=2,
        encode_input=False,
        encode_output=False,
    )
    assert len(train_loader.dataset.chunk_paths) == 2
    assert val_loaders["val"].dataset.chunk_paths[0].name == names[2]
    assert test_loader.dataset.chunk_paths[0].name == names[3]

    with pytest.raises(ValueError, match="must not overlap"):
        load_chunked_multiop_darcy(
            chunk_dir=chunk_dir,
            n_context=2,
            batch_size=2,
            train_chunks=[0, 1],
            val_chunks=[1],
            test_chunks=[2],
            encode_input=False,
            encode_output=False,
        )


def test_dataset_keeps_only_one_cached_chunk(chunk_dir):
    dataset = ChunkedMultiOperatorDarcyDataset(
        [chunk_dir / "chunk_00.pt", chunk_dir / "chunk_01.pt"],
        n_context=2,
        n_samples=4,
        random_context=False,
    )
    dataset[(0, 0)]
    first_cache = dataset._cached_chunk
    dataset[(1, 0)]

    assert dataset.cached_chunk_index == 1
    assert dataset._cached_chunk is not first_cache
    assert dataset.chunk_load_count == 2


def test_num_workers_is_rejected(chunk_dir):
    with pytest.raises(ValueError, match="num_workers=0"):
        _load(chunk_dir, num_workers=1)
