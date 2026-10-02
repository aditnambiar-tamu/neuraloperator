"""Train and evaluate FNOSets on chunked 1D Darcy data."""

from pathlib import Path

try:  # Supports both ``python examples/models/...py`` and package imports.
    from .fnosets_chunked_utils import run_fnosets_chunked_forward_example
except ImportError:
    from fnosets_chunked_utils import run_fnosets_chunked_forward_example


# Update these placeholder paths for your data and checkpoint location.
CHUNK_DIR = "/path/to/poisson1d_dataset_chunks"
CHECKPOINT_DIR = Path("./checkpoints/fnosets_1d_darcy")


if __name__ == "__main__":
    run_fnosets_chunked_forward_example(
        chunk_dir=CHUNK_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        chunk_pattern="poisson1d_chunk_*.pt",
        spatial_dim=1,
        n_modes=(16,),
        n_context=8,
        fixed_context_indices=None,
        batch_size=32,
        test_batch_size=32,
        n_train_chunks=40,
        n_val_chunks=5,
        n_test_chunks=5,
        n_train_samples=160_000,
        n_val_samples=20_000,
        n_test_samples=20_000,
        hidden_channels=256,
        n_epochs=30,
        eval_interval=5,
        periodic_in_x=False,
        title="FNOSets 1D Darcy",
    )
