"""Train and evaluate FNOSets on chunked 2D Darcy data with scalar k."""

from pathlib import Path

try:
    from .fnosets_chunked_utils import run_fnosets_chunked_forward_example
except ImportError:
    from fnosets_chunked_utils import run_fnosets_chunked_forward_example


# Update these placeholder paths for your data and checkpoint location.
CHUNK_DIR = "/path/to/poisson2d_scalar_dataset_chunks"
CHECKPOINT_DIR = Path("./checkpoints/fnosets_2d_scalar_darcy")


if __name__ == "__main__":
    run_fnosets_chunked_forward_example(
        chunk_dir=CHUNK_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        chunk_pattern="poisson2d_chunk_*.pt",
        spatial_dim=2,
        n_modes=(16, 16),
        n_context=8,
        fixed_context_indices=None,
        batch_size=5,
        test_batch_size=5,
        n_train_chunks=40,
        n_val_chunks=5,
        n_test_chunks=5,
        n_train_samples=40_000,
        n_val_samples=5_000,
        n_test_samples=5_000,
        hidden_channels=128,
        n_epochs=30,
        eval_interval=5,
        periodic_in_x=False,
        periodic_in_y=False,
        title="FNOSets 2D Darcy (scalar k)",
    )
