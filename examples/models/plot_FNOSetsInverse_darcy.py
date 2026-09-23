"""
Training FNOSetsInverse on Multi-Operator Darcy/Poisson Data
===========================================================

We train an FNOSetsInverse model to infer the coefficient field ``k`` from
episodic in-context forcing/solution pairs.
"""

import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from neuralop.models import FNOSetsInverse
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.training.training_state import load_training_state
from neuralop.data.datasets import (
    RelativeFrobeniusLoss,
    build_fnosets_inverse_data_processor,
    load_chunked_multiop_darcy,
    save_fnosets_inverse_data_processor,
)
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss


def plot_fnosets_inverse_prediction_1d(
    model,
    dataset,
    data_processor,
    index=0,
    device="cpu",
    title="FNOSetsInverse coefficient prediction",
    save_path=None,
):
    model.eval()
    data_processor.eval()

    sample = dataset[index]
    sample = {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    sample = data_processor.preprocess(sample)

    with torch.no_grad():
        pred = model(**sample)
        pred, sample = data_processor.postprocess(pred, sample)

    pred = pred[0, 0].detach().cpu()
    truth = sample["y"][0, 0].detach().cpu()
    grid = torch.linspace(0, 1, pred.shape[-1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid.tolist(), truth.tolist(), label="Ground truth k", linewidth=2)
    ax.plot(grid.tolist(), pred.tolist(), "--", label="Prediction", linewidth=2)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()
    return fig, ax


def plot_fnosets_inverse_prediction_2d(
    model,
    dataset,
    data_processor,
    index=0,
    device="cpu",
    title="FNOSetsInverse coefficient prediction",
    save_path=None,
):
    model.eval()
    data_processor.eval()

    sample = dataset[index]
    sample = {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    sample = data_processor.preprocess(sample)

    with torch.no_grad():
        pred = model(**sample)
        pred, sample = data_processor.postprocess(pred, sample)

    pred = pred[0].detach().cpu()
    truth = sample["y"][0].detach().cpu()
    component_names = data_processor.coefficient_components
    if pred.shape[0] != len(component_names):
        raise ValueError(
            f"Expected {len(component_names)} coefficient channels, got "
            f"{pred.shape[0]}."
        )

    fig, axes = plt.subplots(
        len(component_names),
        3,
        figsize=(12, 3.5 * len(component_names)),
        squeeze=False,
    )
    for row, component_name in enumerate(component_names):
        component_pred = pred[row]
        component_truth = truth[row]
        vmin = min(component_pred.min().item(), component_truth.min().item())
        vmax = max(component_pred.max().item(), component_truth.max().item())
        truth_image = axes[row, 0].imshow(
            component_truth, origin="lower", vmin=vmin, vmax=vmax
        )
        axes[row, 0].set_title(f"Ground truth {component_name}")
        axes[row, 1].imshow(component_pred, origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Prediction {component_name}")
        error_image = axes[row, 2].imshow(
            (component_pred - component_truth).abs(), origin="lower"
        )
        axes[row, 2].set_title(f"Absolute error {component_name}")
        fig.colorbar(truth_image, ax=axes[row, :2], shrink=0.75)
        fig.colorbar(error_image, ax=axes[row, 2], shrink=0.75)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()
    return fig, axes


def train_chunked_fnosets_inverse(
    chunk_dir,
    *,
    chunk_pattern="poisson1d_chunk_*.pt",
    n_train_chunks=40,
    n_val_chunks=5,
    n_test_chunks=5,
    n_context=8,
    fixed_context_indices=None,
    batch_size=32,
    test_batch_size=32,
    n_train_samples=160_000,
    n_val_samples=20_000,
    n_test_samples=20_000,
    n_modes=(16,),
    hidden_channels=256,
    n_epochs=30,
    eval_interval=5,
    periodic_in_x=True,
    periodic_in_y=True,
    checkpoint_dir="./ckpt/fnosets_inverse_darcy",
    device=None,
):
    """Train scalar or symmetric-tensor FNOSetsInverse on chunked data.

    When contexts are fixed, changing the loader's unused query pair does not
    change an inverse example. Choose validation and test sample counts with
    that duplication in mind.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spatial_dim = len(n_modes)
    checkpoint_dir = Path(checkpoint_dir)

    train_loader, val_loaders, final_test_loader, base_data_processor = (
        load_chunked_multiop_darcy(
            chunk_dir=chunk_dir,
            chunk_pattern=chunk_pattern,
            n_train_chunks=n_train_chunks,
            n_val_chunks=n_val_chunks,
            n_test_chunks=n_test_chunks,
            n_context=n_context,
            fixed_context_indices=fixed_context_indices,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            n_train_samples=n_train_samples,
            n_val_samples=n_val_samples,
            n_test_samples=n_test_samples,
            encode_input=True,
            encode_output=True,
            include_k=True,
            operator_direction="f_to_u",
        )
    )

    # This streams k through memory one training chunk at a time. The inverse
    # processor changes the forward loader's query target to k, so the original
    # chunk-aware loaders and their cache-friendly batch order remain intact.
    data_processor = build_fnosets_inverse_data_processor(
        base_data_processor,
        train_loader.dataset,
    ).to(device)
    if data_processor.spatial_ndim != spatial_dim:
        raise ValueError(
            f"n_modes has {spatial_dim} dimensions, but the dataset has "
            f"{data_processor.spatial_ndim} spatial dimensions."
        )

    model = FNOSetsInverse(
        n_modes=n_modes,
        in_channels=1,
        out_channels=1,
        coefficient_channels=data_processor.coefficient_channels,
        hidden_channels=hidden_channels,
        encoder_layers=4,
        decoder_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        channel_mlp_expansion=2.0,
        enforce_hermitian_symmetry=spatial_dim != 1,
    ).to(device)

    print(f"\nOur model has {count_model_params(model)} parameters.")
    sys.stdout.flush()

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    h1loss = H1Loss(
        d=spatial_dim,
        periodic_in_x=periodic_in_x,
        periodic_in_y=periodic_in_y,
    )
    eval_losses = {
        "h1": h1loss,
        "l2": LpLoss(d=spatial_dim, p=2),
    }
    if data_processor.coefficient_representation == "symmetric_2d":
        eval_losses["frobenius"] = RelativeFrobeniusLoss()
        checkpoint_metric = "val_frobenius"
    else:
        checkpoint_metric = "val_h1"

    trainer = Trainer(
        model=model,
        n_epochs=n_epochs,
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=eval_interval,
        use_distributed=False,
        verbose=True,
        progress_bar=True,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_fnosets_inverse_data_processor(
        data_processor,
        checkpoint_dir / "data_processor.pt",
        metadata={
            "schema_version": 1,
            "spatial_dim": spatial_dim,
            "h1": {
                "periodic_in_x": periodic_in_x,
                "periodic_in_y": periodic_in_y,
            },
            # These settings rebuild the exact splits for evaluation with the
            # saved processor, so forward normalizers need not be fitted again.
            "evaluation_loader": {
                "chunk_pattern": chunk_pattern,
                "train_chunks": [
                    path.name for path in train_loader.dataset.chunk_paths
                ],
                "val_chunks": [
                    path.name for path in val_loaders["val"].dataset.chunk_paths
                ],
                "test_chunks": [
                    path.name for path in final_test_loader.dataset.chunk_paths
                ],
                "n_context": n_context,
                "fixed_context_indices": (
                    None
                    if fixed_context_indices is None
                    else (
                        fixed_context_indices.tolist()
                        if torch.is_tensor(fixed_context_indices)
                        else list(fixed_context_indices)
                    )
                ),
                "batch_size": batch_size,
                "test_batch_size": test_batch_size,
                "n_train_samples": n_train_samples,
                "n_val_samples": n_val_samples,
                "n_test_samples": n_test_samples,
                "include_k": True,
                "operator_direction": "f_to_u",
                "encode_input": False,
                "encode_output": False,
            },
        },
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=val_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=h1loss,
        eval_losses=eval_losses,
        save_best=checkpoint_metric,
        save_dir=checkpoint_dir,
    )

    model, _, _, _, best_epoch = load_training_state(
        save_dir=checkpoint_dir,
        save_name="best_model",
        model=model,
        map_location=device,
    )
    trainer.model = model
    final_metrics = trainer.evaluate(
        eval_losses,
        final_test_loader,
        log_prefix="test",
    )
    print(f"Best validation checkpoint epoch: {best_epoch}")
    print("Final test metrics:", final_metrics)

    return model, data_processor, final_test_loader, final_metrics


if __name__ == "__main__":
    # In Colab, this may point at a directory mounted from Google Drive.
    chunk_dir = os.environ.get("NEURALOP_DARCY_CHUNK_DIR")
    if chunk_dir:
        train_chunked_fnosets_inverse(Path(chunk_dir))
    else:
        print(
            "Set NEURALOP_DARCY_CHUNK_DIR or call "
            "train_chunked_fnosets_inverse(chunk_dir) explicitly."
        )
