import pytest
import torch

from neuralop import H1Loss, Trainer
from neuralop.data.datasets import (
    RelativeFrobeniusLoss,
    build_fnosets_inverse_data_processor,
    channels_to_symmetric_tensor,
    fit_coefficient_normalizer,
    load_chunked_multiop_darcy,
    load_fnosets_inverse_data_processor,
    relative_frobenius_error,
    save_fnosets_inverse_data_processor,
    smallest_symmetric_eigenvalue,
    symmetric_tensor_to_channels,
    wrap_fnosets_inverse_loader,
)
from neuralop.models import FNOSetsInverse


def _make_spd_coefficients(value, n_operators=2, spatial_shape=(5, 6)):
    height, width = spatial_shape
    operator = torch.arange(n_operators, dtype=torch.float32)[:, None, None]
    row = torch.arange(height, dtype=torch.float32)[None, :, None]
    column = torch.arange(width, dtype=torch.float32)[None, None, :]

    k11 = 2.0 + 0.4 * value + 0.2 * operator + 0.03 * row + 0.02 * column
    k22 = 3.0 + 0.3 * value + 0.1 * operator + 0.02 * row + 0.04 * column
    k12 = 0.15 + 0.02 * value + 0.01 * operator + 0.002 * row - 0.001 * column

    coefficient = torch.empty(n_operators, height, width, 2, 2)
    coefficient[..., 0, 0] = k11
    coefficient[..., 1, 1] = k22
    coefficient[..., 0, 1] = k12
    coefficient[..., 1, 0] = k12

    determinants = k11 * k22 - k12.square()
    assert torch.all(k11 > 0)
    assert torch.all(determinants > 0)
    return coefficient


def _write_spd_chunk(path, value, n_operators=2, n_pairs=4, spatial_shape=(5, 6)):
    height, width = spatial_shape
    operator = torch.arange(n_operators, dtype=torch.float32)[:, None, None, None]
    pair = torch.arange(n_pairs, dtype=torch.float32)[None, :, None, None]
    row = torch.arange(height, dtype=torch.float32)[None, None, :, None]
    column = torch.arange(width, dtype=torch.float32)[None, None, None, :]
    forcing = value + 0.4 * operator + 0.2 * pair + 0.03 * row + 0.01 * column

    torch.save(
        {
            "k": _make_spd_coefficients(value, n_operators, spatial_shape),
            "f": forcing,
            "u": 1.5 * forcing + 2.0,
        },
        path,
    )


@pytest.fixture
def tensor_chunk_dir(tmp_path):
    for index in range(4):
        _write_spd_chunk(tmp_path / f"tensor_chunk_{index}.pt", float(index))
    return tmp_path


def _load_tensor_chunks(chunk_dir, **kwargs):
    options = {
        "chunk_dir": chunk_dir,
        "chunk_pattern": "tensor_chunk_*.pt",
        "n_context": 2,
        "batch_size": 2,
        "test_batch_size": 2,
        "n_train_samples": 4,
        "n_val_samples": 2,
        "n_test_samples": 2,
        "n_train_chunks": 2,
        "n_val_chunks": 1,
        "n_test_chunks": 1,
        "encode_input": True,
        "encode_output": True,
        "include_k": True,
        "seed": 0,
    }
    options.update(kwargs)
    return load_chunked_multiop_darcy(**options)


def _manual_symmetric_channels(coefficient):
    return torch.stack(
        (
            coefficient[..., 0, 0],
            coefficient[..., 1, 1],
            coefficient[..., 0, 1],
        ),
        dim=1,
    )


def _make_tensor_inverse_model():
    return FNOSetsInverse(
        n_modes=(2, 2),
        in_channels=1,
        out_channels=1,
        coefficient_channels=3,
        hidden_channels=4,
        encoder_layers=1,
        decoder_layers=1,
        lifting_channel_ratio=1,
        projection_channel_ratio=1,
        channel_mlp_expansion=1.0,
        enforce_hermitian_symmetry=False,
    )


def test_symmetric_tensor_channel_conversion_round_trip():
    coefficient = _make_spd_coefficients(value=0.5)
    expected_channels = _manual_symmetric_channels(coefficient)

    channels = symmetric_tensor_to_channels(coefficient)
    reconstructed = channels_to_symmetric_tensor(channels)

    assert channels.shape == (2, 3, 5, 6)
    assert torch.equal(channels, expected_channels)
    assert torch.equal(reconstructed, coefficient)


def test_symmetric_tensor_conversion_rejects_nonsymmetric_input():
    coefficient = _make_spd_coefficients(value=0.5)
    coefficient[..., 1, 0] += 0.25

    with pytest.raises(ValueError, match="symmetric"):
        symmetric_tensor_to_channels(coefficient)


def test_symmetric_tensor_conversion_averages_roundoff_level_asymmetry():
    coefficient = _make_spd_coefficients(value=0.5)
    coefficient[..., 0, 1] += 2.5e-7
    coefficient[..., 1, 0] -= 2.5e-7
    expected_off_diagonal = 0.5 * (coefficient[..., 0, 1] + coefficient[..., 1, 0])

    channels = symmetric_tensor_to_channels(coefficient)

    assert torch.equal(channels[:, 2], expected_off_diagonal)


def test_relative_frobenius_error_matches_reconstructed_matrix_norm():
    target = symmetric_tensor_to_channels(_make_spd_coefficients(value=0.5))
    prediction = target.clone()
    prediction[:, 0] += 0.1
    prediction[:, 2] -= 0.2

    actual = relative_frobenius_error(prediction, target, reduction="none")
    prediction_matrix = channels_to_symmetric_tensor(prediction)
    target_matrix = channels_to_symmetric_tensor(target)
    expected = torch.linalg.vector_norm(
        (prediction_matrix - target_matrix).flatten(1), dim=1
    ) / torch.linalg.vector_norm(target_matrix.flatten(1), dim=1)

    assert torch.allclose(actual, expected)
    assert torch.allclose(RelativeFrobeniusLoss()(prediction, target), expected.sum())


def test_smallest_symmetric_eigenvalue_matches_torch():
    coefficient = _make_spd_coefficients(value=0.5)
    channels = symmetric_tensor_to_channels(coefficient)

    actual = smallest_symmetric_eigenvalue(channels)
    expected = torch.linalg.eigvalsh(coefficient)[..., 0]

    assert torch.allclose(actual, expected)

    indefinite = torch.zeros(1, 3, 2, 2)
    indefinite[:, 0] = 1
    indefinite[:, 1] = 1
    indefinite[:, 2] = 2
    assert torch.all(smallest_symmetric_eigenvalue(indefinite) == -1)


def test_tensor_coefficient_normalizer_fits_each_channel_from_train_chunks(
    tensor_chunk_dir,
):
    train_loader, _, _, _ = _load_tensor_chunks(tensor_chunk_dir)

    normalizer = fit_coefficient_normalizer(
        train_loader.dataset,
        coefficient_representation="symmetric_2d",
    )
    train_coefficients = torch.cat(
        [
            torch.load(path, weights_only=False)["k"]
            for path in train_loader.dataset.chunk_paths
        ]
    )
    expected_channels = _manual_symmetric_channels(train_coefficients)

    assert normalizer.mean.shape == (1, 3, 1, 1)
    assert normalizer.std.shape == (1, 3, 1, 1)
    assert normalizer.dim == [0, 2, 3]
    assert torch.allclose(
        normalizer.mean,
        expected_channels.mean(dim=[0, 2, 3], keepdim=True),
    )
    assert torch.allclose(
        normalizer.std,
        expected_channels.std(dim=[0, 2, 3], keepdim=True),
    )


def test_tensor_inverse_loader_wrapper_encodes_coefficient_target(tensor_chunk_dir):
    train_loader, _, _, _ = _load_tensor_chunks(tensor_chunk_dir)

    wrapped = wrap_fnosets_inverse_loader(train_loader)
    batch = next(iter(wrapped))

    assert wrapped.batch_sampler is train_loader.batch_sampler
    assert batch["k"].shape == (2, 1, 5, 6, 2, 2)
    assert batch["y"].shape == (2, 3, 5, 6)
    assert torch.equal(
        batch["y"],
        _manual_symmetric_channels(batch["k"][:, 0]),
    )

    with pytest.raises(ValueError, match="already wrapped"):
        wrap_fnosets_inverse_loader(
            wrapped,
            coefficient_representation="scalar",
        )


def test_tensor_inverse_processor_and_checkpoint_round_trip(tensor_chunk_dir, tmp_path):
    train_loader, _, _, base_processor = _load_tensor_chunks(tensor_chunk_dir)
    processor = build_fnosets_inverse_data_processor(
        base_processor,
        train_loader.dataset,
    )
    item_coefficient = train_loader.dataset[0]["k"]
    item_target = processor.encode_coefficient(
        item_coefficient,
        batched=False,
    )
    assert item_target.shape == (3, 5, 6)
    batch = next(iter(train_loader))
    raw_coefficient = batch["k"].clone()
    expected_target = _manual_symmetric_channels(raw_coefficient[:, 0])

    processor.train()
    training_processed = processor.preprocess(
        {key: value.clone() for key, value in batch.items()}
    )
    assert torch.allclose(
        training_processed["y"],
        processor.coefficient_normalizer.transform(expected_target),
    )

    processor.eval()
    processed = processor.preprocess(
        {key: value.clone() for key, value in batch.items()}
    )
    assert processor.coefficient_representation == "symmetric_2d"
    assert processor.coefficient_channels == 3
    assert processor.coefficient_components == ("K11", "K22", "K12")
    assert processed["y"].shape == (2, 3, 5, 6)
    assert torch.equal(processed["y"], expected_target)
    assert torch.equal(
        channels_to_symmetric_tensor(processed["y"]), raw_coefficient[:, 0]
    )
    normalized_prediction = processor.coefficient_normalizer.transform(
        expected_target
    )
    physical_prediction, _ = processor.postprocess(
        normalized_prediction,
        processed,
    )
    assert torch.allclose(physical_prediction, expected_target)
    assert relative_frobenius_error(physical_prediction, processed["y"]) == 0

    checkpoint_path = tmp_path / "tensor_inverse_processor.pt"
    save_fnosets_inverse_data_processor(
        processor,
        checkpoint_path,
        metadata={"purpose": "tensor-test"},
    )
    loaded, metadata = load_fnosets_inverse_data_processor(checkpoint_path)
    loaded.eval()
    loaded_processed = loaded.preprocess(
        {key: value.clone() for key, value in batch.items()}
    )

    assert metadata == {"purpose": "tensor-test"}
    assert loaded.coefficient_representation == "symmetric_2d"
    assert torch.equal(loaded_processed["y"], processed["y"])
    assert torch.equal(
        loaded.coefficient_normalizer.mean,
        processor.coefficient_normalizer.mean,
    )
    assert torch.equal(
        loaded.coefficient_normalizer.std,
        processor.coefficient_normalizer.std,
    )


def test_legacy_inverse_processor_checkpoint_defaults_to_scalar(tmp_path):
    checkpoint_path = tmp_path / "legacy_scalar_processor.pt"
    torch.save(
        {
            "input_normalizer": None,
            "output_normalizer": None,
            "coefficient_normalizer": None,
            "metadata": {"legacy": True},
        },
        checkpoint_path,
    )

    processor, metadata = load_fnosets_inverse_data_processor(checkpoint_path)

    assert processor.coefficient_representation == "scalar"
    assert processor.coefficient_channels == 1
    assert metadata == {"legacy": True}


def test_processor_checkpoint_rejects_inconsistent_tensor_normalizer(tmp_path):
    checkpoint_path = tmp_path / "inconsistent_tensor_processor.pt"
    torch.save(
        {
            "input_normalizer": None,
            "output_normalizer": None,
            "coefficient_normalizer": {
                "mean": torch.zeros(1, 1, 1, 1),
                "std": torch.ones(1, 1, 1, 1),
                "eps": 1e-7,
                "dim": [0, 2, 3],
            },
            "coefficient_representation": "symmetric_2d",
            "spatial_ndim": 2,
            "metadata": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="incompatible shape"):
        load_fnosets_inverse_data_processor(checkpoint_path)


def test_tensor_inverse_forward_and_per_component_h1_backward(tensor_chunk_dir):
    train_loader, _, _, base_processor = _load_tensor_chunks(tensor_chunk_dir)
    processor = build_fnosets_inverse_data_processor(
        base_processor,
        train_loader.dataset,
        coefficient_representation="symmetric_2d",
    )
    processor.train()
    batch = processor.preprocess(next(iter(train_loader)))
    model = _make_tensor_inverse_model()

    prediction = model(**batch)
    loss_function = H1Loss(
        d=2,
        reduction="sum",
        periodic_in_x=False,
        periodic_in_y=False,
    )
    loss = loss_function(prediction, batch["y"])
    component_loss = sum(
        loss_function(
            prediction[:, channel : channel + 1],
            batch["y"][:, channel : channel + 1],
        )
        for channel in range(3)
    )
    loss.backward()

    assert prediction.shape == (2, 3, 5, 6)
    assert torch.isfinite(loss)
    assert torch.allclose(loss.detach(), component_loss.detach())
    assert any(parameter.grad is not None for parameter in model.parameters())

    wrong_target_batch = dict(batch)
    wrong_target_batch["y"] = batch["y"][:, :1]
    with pytest.raises(ValueError, match="target channels"):
        model(**wrong_target_batch)

    wrong_spatial_batch = dict(batch)
    wrong_spatial_batch["y"] = batch["y"][..., :-1]
    with pytest.raises(ValueError, match="output and target shapes"):
        model(**wrong_spatial_batch)


def test_tensor_inverse_context_only_inference(tensor_chunk_dir):
    train_loader, _, _, base_processor = _load_tensor_chunks(tensor_chunk_dir)
    processor = build_fnosets_inverse_data_processor(
        base_processor,
        train_loader.dataset,
    )
    processor.eval()
    context_batch = next(iter(train_loader))
    context_batch.pop("k")
    context_batch.pop("y")

    processed = processor.preprocess(context_batch)
    with torch.no_grad():
        prediction = _make_tensor_inverse_model()(**processed)
        prediction, processed = processor.postprocess(prediction, processed)

    assert "y" not in processed
    assert prediction.shape == (2, 3, 5, 6)
    assert processor.decode_coefficient(prediction).shape == (2, 5, 6, 2, 2)


def test_tensor_inverse_trainer_evaluates_physical_metrics(tensor_chunk_dir):
    train_loader, val_loaders, _, base_processor = _load_tensor_chunks(
        tensor_chunk_dir
    )
    processor = build_fnosets_inverse_data_processor(
        base_processor,
        train_loader.dataset,
    )
    trainer = Trainer(
        model=_make_tensor_inverse_model(),
        n_epochs=1,
        device="cpu",
        data_processor=processor,
    )
    metrics = trainer.evaluate(
        {
            "h1": H1Loss(
                d=2,
                periodic_in_x=False,
                periodic_in_y=False,
            ),
            "frobenius": RelativeFrobeniusLoss(),
        },
        val_loaders["val"],
        log_prefix="val",
    )

    assert set(metrics) == {"val_h1", "val_frobenius"}
    assert all(torch.isfinite(torch.as_tensor(value)) for value in metrics.values())


def test_tensor_inverse_processor_rejects_model_target_shape_mismatch(
    tensor_chunk_dir,
):
    train_loader, _, _, base_processor = _load_tensor_chunks(tensor_chunk_dir)
    processor = build_fnosets_inverse_data_processor(
        base_processor,
        train_loader.dataset,
    )
    processor.train()
    batch = processor.preprocess(next(iter(train_loader)))
    one_channel_prediction = torch.zeros(
        batch["y"].shape[0],
        1,
        *batch["y"].shape[2:],
    )

    with pytest.raises(ValueError, match="output channels"):
        processor.postprocess(one_channel_prediction, batch)
