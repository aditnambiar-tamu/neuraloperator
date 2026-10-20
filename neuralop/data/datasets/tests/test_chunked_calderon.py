from pathlib import Path

import pytest
import torch

from neuralop.data.datasets import (
    ChunkedCalderonDataset,
    load_calderon_data_processor,
    load_chunked_calderon,
    save_calderon_data_processor,
)
from neuralop.losses import H1Loss
from neuralop.models import FNOSetsCalderon


def _write_chunk(path, value, n_operators=2, n_examples=4, n_boundary_points=8):
    operator_values = torch.arange(n_operators, dtype=torch.float32).reshape(-1, 1, 1)
    example_values = torch.arange(n_examples, dtype=torch.float32).reshape(1, -1, 1)
    f_boundary = value + 10 * operator_values + example_values
    f_boundary = f_boundary.expand(n_operators, n_examples, n_boundary_points).clone()
    n_boundary = f_boundary + 100

    k = value + torch.arange(n_operators, dtype=torch.float32).reshape(-1, 1, 1)
    k = k.expand(n_operators, 5, 7).clone()
    torch.save(
        {"k": k, "f_boundary": f_boundary, "n_boundary": n_boundary},
        path,
    )


@pytest.fixture
def calderon_chunks(tmp_path):
    for index in (2, 0, 1):
        _write_chunk(tmp_path / f"calderon_chunk_{index:02d}.pt", index * 100)
    torch.save({"chunks": 3}, tmp_path / "manifest.pt")
    return tmp_path


def _load(chunk_dir, **overrides):
    options = {
        "chunk_dir": chunk_dir,
        "chunk_pattern": "calderon_chunk_*.pt",
        "n_context": 2,
        "batch_size": 2,
        "n_train_chunks": 1,
        "n_val_chunks": 1,
        "n_test_chunks": 1,
        "n_train_samples": 4,
        "n_val_samples": 2,
        "n_test_samples": 2,
    }
    options.update(overrides)
    return load_chunked_calderon(**options)


def test_loader_splits_chunks_and_returns_paired_contexts(calderon_chunks):
    train_loader, val_loaders, test_loader, processor = _load(calderon_chunks)

    assert [path.name for path in train_loader.dataset.chunk_paths] == [
        "calderon_chunk_00.pt"
    ]
    assert [path.name for path in val_loaders["val"].dataset.chunk_paths] == [
        "calderon_chunk_01.pt"
    ]
    assert [path.name for path in test_loader.dataset.chunk_paths] == [
        "calderon_chunk_02.pt"
    ]

    raw = train_loader.dataset[0]
    assert raw["f_boundary_context"].shape == (2, 1, 8)
    assert raw["n_boundary_context"].shape == (2, 1, 8)
    assert raw["y"].shape == (1, 5, 7)
    assert torch.allclose(
        raw["n_boundary_context"] - raw["f_boundary_context"],
        torch.full_like(raw["f_boundary_context"], 100),
    )

    batch = next(iter(train_loader))
    assert batch["f_boundary_context"].shape == (2, 2, 1, 8)
    assert batch["n_boundary_context"].shape == (2, 2, 1, 8)
    assert batch["y"].shape == (2, 1, 5, 7)
    assert train_loader.dataset.chunk_load_count == 1
    assert processor.f_boundary_normalizer is not None


def test_normalizers_use_only_training_chunks_and_round_trip(tmp_path, calderon_chunks):
    _, _, _, processor = _load(calderon_chunks)
    # The training chunk has operator/example values 0..13; validation and
    # test chunks have much larger values and must not enter these statistics.
    assert processor.f_boundary_normalizer.mean.item() == pytest.approx(6.5)
    assert processor.n_boundary_normalizer.mean.item() == pytest.approx(106.5)
    assert processor.coefficient_normalizer.mean.item() == pytest.approx(0.5)

    processor.eval()
    target = torch.randn(2, 1, 5, 7)
    data_dict = {"y": target.clone()}
    normalized = processor.coefficient_normalizer.transform(target)
    restored, _ = processor.postprocess(normalized, data_dict)
    assert torch.allclose(restored, target, atol=1e-6)

    checkpoint = tmp_path / "processor.pt"
    save_calderon_data_processor(processor, checkpoint, metadata={"n_context": 2})
    loaded, metadata = load_calderon_data_processor(checkpoint)
    assert metadata == {"n_context": 2}
    assert loaded.f_boundary_normalizer.mean.item() == pytest.approx(6.5)
    assert loaded.coefficient_normalizer.std.item() == pytest.approx(
        processor.coefficient_normalizer.std.item()
    )


def test_all_boundary_examples_can_be_context_and_fixed_context_is_supported(
    calderon_chunks,
):
    dataset = ChunkedCalderonDataset(
        [calderon_chunks / "calderon_chunk_00.pt"],
        n_context=4,
        fixed_context_indices=[0, 1, 2, 3],
    )
    sample = dataset[0]
    assert sample["f_boundary_context"].shape == (4, 1, 8)
    assert sample["n_boundary_context"].shape == (4, 1, 8)


def test_chunked_calderon_rejects_unpaired_boundary_arrays(tmp_path):
    path = tmp_path / "bad_chunk.pt"
    _write_chunk(path, 0)
    chunk = torch.load(path, weights_only=False)
    chunk["n_boundary"] = chunk["n_boundary"][:, :, :-1]
    torch.save(chunk, path)

    with pytest.raises(ValueError, match="identical shapes"):
        ChunkedCalderonDataset([path], n_context=2)


def test_chunked_calderon_requires_zero_workers(calderon_chunks):
    with pytest.raises(ValueError, match="num_workers=0"):
        _load(calderon_chunks, num_workers=1)


def test_loader_processor_model_and_loss_work_together(calderon_chunks):
    train_loader, val_loaders, _, processor = _load(calderon_chunks)
    model = FNOSetsCalderon(
        boundary_n_modes=(4,),
        coefficient_n_modes=(4, 4),
        hidden_channels=8,
        encoder_layers=1,
        decoder_layers=1,
        channel_mlp_expansion=1.0,
    )
    h1_loss = H1Loss(d=2, periodic_in_x=False, periodic_in_y=False)

    processor.train()
    train_batch = processor.preprocess(next(iter(train_loader)))
    prediction = model(**train_batch)
    training_loss = h1_loss(prediction, **train_batch)
    training_loss.backward()
    assert torch.isfinite(training_loss)

    processor.eval()
    val_batch = processor.preprocess(next(iter(val_loaders["val"])))
    val_prediction = model(**val_batch)
    val_prediction, val_batch = processor.postprocess(val_prediction, val_batch)
    assert val_prediction.shape == val_batch["y"].shape
    assert torch.isfinite(h1_loss(val_prediction, **val_batch))
