"""Shared training, evaluation, and plotting helpers for chunked FNOSets examples.

The runnable examples live beside this module. Each example supplies its
chunk/checkpoint paths and model settings, while these helpers keep the
forward and inverse workflows consistent.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from neuralop import H1Loss, LpLoss, Trainer
from neuralop.data.datasets import (
    RelativeFrobeniusLoss,
    build_fnosets_inverse_data_processor,
    load_chunked_multiop_darcy,
    save_fnosets_inverse_data_processor,
)
from neuralop.models import FNOSets, FNOSetsInverse
from neuralop.training import AdamW
from neuralop.training.training_state import load_training_state
from neuralop.utils import count_model_params


def _sample_indices(dataset, n_examples):
    if len(dataset) < 1:
        raise ValueError("The test dataset is empty.")
    if n_examples < 1:
        raise ValueError("n_examples must be at least 1.")
    return random.sample(range(len(dataset)), k=min(n_examples, len(dataset)))


def _predict_sample(model, dataset, data_processor, index, device):
    """Run one unbatched dataset item through a processor and model."""

    raw_sample = dataset[index]
    # Keep the physical query function for visualization. The processor below
    # normalizes a separate batched copy for the model.
    query = raw_sample["u_query"][0].detach().cpu()
    sample = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in raw_sample.items()
    }
    sample = data_processor.preprocess(sample, batched=True)
    model_inputs = {
        key: sample[key] for key in ("u_context", "f_context", "u_query")
    }

    with torch.no_grad():
        prediction = model(**model_inputs)
        prediction, sample = data_processor.postprocess(prediction, sample)

    prediction = prediction[0].detach().cpu()
    truth = sample["y"][0].detach().cpu()
    return query, prediction, truth


def plot_random_forward_examples(
    model,
    test_loader,
    data_processor,
    *,
    spatial_dim,
    n_examples=5,
    title="FNOSets test predictions",
):
    """Plot query inputs, true outputs, predictions, and absolute errors."""

    model.eval()
    data_processor.eval()
    indices = _sample_indices(test_loader.dataset, n_examples)

    for example_number, index in enumerate(indices, start=1):
        query, prediction, truth = _predict_sample(
            model, test_loader.dataset, data_processor, index, data_processor.device
        )
        # Forward Darcy examples have one scalar output channel.
        query = query[0]
        prediction = prediction[0]
        truth = truth[0]
        error = (prediction - truth).abs()

        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        fig.suptitle(f"{title}: example {example_number} (dataset index {index})")
        if spatial_dim == 1:
            grid = torch.linspace(0, 1, query.shape[-1])
            axes[0].plot(grid, query)
            axes[0].set_title("Query forcing f")
            axes[1].plot(grid, truth)
            axes[1].set_title("True solution u")
            axes[2].plot(grid, prediction)
            axes[2].set_title("Predicted solution u")
            axes[3].plot(grid, error)
            axes[3].set_title("Absolute error")
            axes[0].set_xlabel("x")
            for ax in axes:
                ax.grid(alpha=0.2)
        else:
            vmin = min(truth.min().item(), prediction.min().item())
            vmax = max(truth.max().item(), prediction.max().item())
            axes[0].imshow(query, origin="lower")
            axes[0].set_title("Query forcing f")
            truth_image = axes[1].imshow(
                truth, origin="lower", vmin=vmin, vmax=vmax
            )
            axes[1].set_title("True solution u")
            axes[2].imshow(prediction, origin="lower", vmin=vmin, vmax=vmax)
            axes[2].set_title("Predicted solution u")
            error_image = axes[3].imshow(error, origin="lower")
            axes[3].set_title("Absolute error")
            fig.colorbar(truth_image, ax=axes[1:3], shrink=0.8)
            fig.colorbar(error_image, ax=axes[3], shrink=0.8)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_random_inverse_examples(
    model,
    test_loader,
    data_processor,
    *,
    spatial_dim,
    n_examples=5,
    title="FNOSetsInverse test predictions",
):
    """Plot true and predicted coefficient components with absolute errors."""

    model.eval()
    data_processor.eval()
    indices = _sample_indices(test_loader.dataset, n_examples)
    component_names = data_processor.coefficient_components

    for example_number, index in enumerate(indices, start=1):
        _, prediction, truth = _predict_sample(
            model, test_loader.dataset, data_processor, index, data_processor.device
        )
        fig, axes = plt.subplots(
            len(component_names),
            3,
            figsize=(12, 3.6 * len(component_names)),
            squeeze=False,
        )
        fig.suptitle(f"{title}: example {example_number} (dataset index {index})")

        for row, component_name in enumerate(component_names):
            pred_component = prediction[row]
            true_component = truth[row]
            error = (pred_component - true_component).abs()

            if spatial_dim == 1:
                grid = torch.linspace(0, 1, pred_component.shape[-1])
                axes[row, 0].plot(grid, true_component)
                axes[row, 1].plot(grid, pred_component)
                axes[row, 2].plot(grid, error)
                axes[row, 0].set_title(f"True {component_name}")
                axes[row, 1].set_title(f"Predicted {component_name}")
                axes[row, 2].set_title(f"Absolute error {component_name}")
                for ax in axes[row]:
                    ax.set_xlabel("x")
                    ax.grid(alpha=0.2)
            else:
                vmin = min(true_component.min().item(), pred_component.min().item())
                vmax = max(true_component.max().item(), pred_component.max().item())
                true_image = axes[row, 0].imshow(
                    true_component, origin="lower", vmin=vmin, vmax=vmax
                )
                axes[row, 0].set_title(f"True {component_name}")
                axes[row, 1].imshow(
                    pred_component, origin="lower", vmin=vmin, vmax=vmax
                )
                axes[row, 1].set_title(f"Predicted {component_name}")
                error_image = axes[row, 2].imshow(error, origin="lower")
                axes[row, 2].set_title(f"Absolute error {component_name}")
                fig.colorbar(true_image, ax=axes[row, :2], shrink=0.8)
                fig.colorbar(error_image, ax=axes[row, 2], shrink=0.8)
                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])

        fig.tight_layout()
        plt.show()
        plt.close(fig)


def run_fnosets_chunked_forward_example(
    *,
    chunk_dir,
    checkpoint_dir,
    chunk_pattern,
    spatial_dim,
    n_modes,
    n_context,
    fixed_context_indices,
    batch_size,
    test_batch_size,
    n_train_chunks,
    n_val_chunks,
    n_test_chunks,
    n_train_samples,
    n_val_samples,
    n_test_samples,
    hidden_channels,
    n_epochs,
    eval_interval,
    periodic_in_x=False,
    periodic_in_y=False,
    n_plot_examples=5,
    title="FNOSets",
    device=None,
):
    """Train, test, and plot a forward FNOSets model on chunked data."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(checkpoint_dir)

    print("Loading chunked Darcy data...")
    train_loader, val_loaders, test_loader, data_processor = (
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
            include_k=False,
            operator_direction="f_to_u",
        )
    )
    data_processor = data_processor.to(device)

    print("Creating FNOSets model...")
    model = FNOSets(
        n_modes=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        encoder_layers=4,
        decoder_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        channel_mlp_expansion=2.0,
        enforce_hermitian_symmetry=spatial_dim != 1,
    ).to(device)
    print(f"Model parameters: {count_model_params(model):,}")

    n_epochs = int(n_epochs)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    h1loss = H1Loss(
        d=spatial_dim,
        periodic_in_x=periodic_in_x,
        periodic_in_y=periodic_in_y,
    )
    eval_losses = {"h1": h1loss, "l2": LpLoss(d=spatial_dim, p=2)}

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
    print("Training FNOSets...")
    trainer.train(
        train_loader=train_loader,
        test_loaders=val_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=h1loss,
        eval_losses=eval_losses,
        save_best="val_h1",
        save_dir=checkpoint_dir,
    )

    model, _, _, _, best_epoch = load_training_state(
        save_dir=checkpoint_dir,
        save_name="best_model",
        model=model,
        map_location=device,
    )
    trainer.model = model
    print(f"Best validation checkpoint: epoch {best_epoch + 1}")
    test_metrics = trainer.evaluate(
        eval_losses, test_loader, log_prefix="test"
    )
    print("Holdout test metrics:", test_metrics)

    plot_random_forward_examples(
        model,
        test_loader,
        data_processor,
        spatial_dim=spatial_dim,
        n_examples=n_plot_examples,
        title=title,
    )
    return model, data_processor, test_loader, test_metrics


def _context_indices_for_metadata(indices):
    if indices is None:
        return None
    if torch.is_tensor(indices):
        return indices.detach().cpu().tolist()
    return list(indices)


def run_fnosets_chunked_inverse_example(
    *,
    chunk_dir,
    checkpoint_dir,
    chunk_pattern,
    spatial_dim,
    n_modes,
    coefficient_representation,
    n_context,
    fixed_context_indices,
    batch_size,
    test_batch_size,
    n_train_chunks,
    n_val_chunks,
    n_test_chunks,
    n_train_samples,
    n_val_samples,
    n_test_samples,
    hidden_channels,
    n_epochs,
    eval_interval,
    periodic_in_x=False,
    periodic_in_y=False,
    n_plot_examples=5,
    title="FNOSetsInverse",
    device=None,
):
    """Train, test, and plot a scalar or symmetric-tensor inverse model."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(checkpoint_dir)

    print("Loading chunked Darcy data...")
    train_loader, val_loaders, test_loader, base_data_processor = (
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
    data_processor = build_fnosets_inverse_data_processor(
        base_data_processor,
        train_loader.dataset,
        coefficient_representation=coefficient_representation,
    ).to(device)
    if data_processor.spatial_ndim != spatial_dim:
        raise ValueError(
            f"Dataset is {data_processor.spatial_ndim}D, but this script "
            f"expects {spatial_dim}D data."
        )

    print("Creating FNOSetsInverse model...")
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
    print(f"Model parameters: {count_model_params(model):,}")

    n_epochs = int(n_epochs)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    h1loss = H1Loss(
        d=spatial_dim,
        periodic_in_x=periodic_in_x,
        periodic_in_y=periodic_in_y,
    )
    eval_losses = {"h1": h1loss, "l2": LpLoss(d=spatial_dim, p=2)}
    if coefficient_representation == "symmetric_2d":
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
            "evaluation_loader": {
                "chunk_pattern": chunk_pattern,
                "train_chunks": [
                    path.name for path in train_loader.dataset.chunk_paths
                ],
                "val_chunks": [
                    path.name for path in val_loaders["val"].dataset.chunk_paths
                ],
                "test_chunks": [path.name for path in test_loader.dataset.chunk_paths],
                "n_context": n_context,
                "fixed_context_indices": _context_indices_for_metadata(
                    fixed_context_indices
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

    print("Training FNOSetsInverse...")
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
    print(f"Best validation checkpoint: epoch {best_epoch + 1}")
    test_metrics = trainer.evaluate(
        eval_losses, test_loader, log_prefix="test"
    )
    print("Holdout test metrics:", test_metrics)

    plot_random_inverse_examples(
        model,
        test_loader,
        data_processor,
        spatial_dim=spatial_dim,
        n_examples=n_plot_examples,
        title=title,
    )
    return model, data_processor, test_loader, test_metrics
