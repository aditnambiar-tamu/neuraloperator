"""Train and evaluate tensor-k FNOSetsInverse on chunked 2D Darcy data."""

from pathlib import Path

try:
    from .fnosets_chunked_utils import run_fnosets_chunked_inverse_example
except ImportError:
    from fnosets_chunked_utils import run_fnosets_chunked_inverse_example


# Update these placeholder paths for your data and checkpoint location.
CHUNK_DIR = "/path/to/poisson2d_tensor_dataset_chunks"
CHECKPOINT_DIR = Path("./checkpoints/fnosets_inverse_2d_tensor_darcy")


if __name__ == "__main__":
    # The model predicts channel-first (K11, K22, K12). The processor converts
    # the stored symmetric matrices to these channels and validates symmetry.
    # Direct channel prediction enforces symmetry on reconstruction but does
    # not itself guarantee positive definiteness.
    run_fnosets_chunked_inverse_example(
        chunk_dir=CHUNK_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        chunk_pattern="poisson2d_chunk_*.pt",
        spatial_dim=2,
        n_modes=(16, 16),
        coefficient_representation="symmetric_2d",
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
        title="FNOSetsInverse 2D Darcy (tensor k)",
    )
