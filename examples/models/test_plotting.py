from pathlib import Path

import torch
import random
import matplotlib.pyplot as plt

from neuralop.models import FNOSets
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training.training_state import load_training_state
from neuralop.data.datasets import load_multiop_darcy
from torch.utils.data import DataLoader
from neuralop.data.datasets import MultiOperatorDarcyDataset

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Plotting a Prediction
# ---------------------
# We plot a query prediction from the held-out test operator split. Passing
# ``test_loaders["test"].dataset`` ensures the sampled episode comes from
# operators that were not used by the training loader.

def _predict_fnosets_sample(model, dataset, data_processor, index=0, device="cpu"):
    """Run FNOSets on one episodic dataset sample."""
    model.eval()
    data_processor.eval()

    sample = dataset[index]
    sample = {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    sample = data_processor.preprocess(sample)

    with torch.no_grad():
        out = model(
            u_context=sample["u_context"],
            f_context=sample["f_context"],
            u_query=sample["u_query"],
        )
        out, sample = data_processor.postprocess(out, sample)

    return out[0, 0].detach().cpu(), sample["y"][0, 0].detach().cpu()


def _relative_errors(pred, truth, d):
    pred_batch = pred.unsqueeze(0).unsqueeze(0)
    truth_batch = truth.unsqueeze(0).unsqueeze(0)
    h1 = H1Loss(d=d)(pred_batch, truth_batch).item()
    l2 = LpLoss(d=d, p=2)(pred_batch, truth_batch).item()
    return {"h1": h1, "l2": l2}


def plot_fnosets_prediction_1d(
    model,
    dataset,
    data_processor,
    index=0,
    device="cpu",
    title="FNOSets query prediction",
    save_path=None,
    print_errors=True,
):
    """Plot 1D FNOSets prediction and ground truth for one episodic sample."""
    pred, truth = _predict_fnosets_sample(
        model=model,
        dataset=dataset,
        data_processor=data_processor,
        index=index,
        device=device,
    )
    errors = _relative_errors(pred, truth, d=1)
    grid = torch.linspace(0, 1, pred.shape[-1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid.tolist(), truth.tolist(), label="Ground truth", linewidth=2)
    ax.plot(grid.tolist(), pred.tolist(), "--", label="Prediction", linewidth=2)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    if print_errors:
        print(f"Plotted sample {index}: relative H1={errors['h1']:.4f}, relative L2={errors['l2']:.4f}")
    plt.show()

    return fig, ax, errors


def plot_fnosets_prediction_2d(
    model,
    dataset,
    data_processor,
    index=0,
    device="cpu",
    title="FNOSets query prediction",
    save_path=None,
    print_errors=True,
    plot_k=False,
):
    """Plot a 2D prediction, optionally including a scalar coefficient field."""
    coefficient = None
    if plot_k:
        raw_sample = dataset[index]
        if "k" not in raw_sample:
            raise ValueError(
                "plot_k=True requires a dataset created with include_k=True."
            )
        coefficient = raw_sample["k"]
        if coefficient.ndim != 3 or coefficient.shape[0] != 1:
            raise ValueError(
                "plot_k=True supports scalar 2D coefficient fields with shape "
                f"(1, H, W), got {tuple(coefficient.shape)}. Set plot_k=False "
                "for tensor-valued coefficients."
            )
        coefficient = coefficient[0].detach().cpu()

    pred, truth = _predict_fnosets_sample(
        model=model,
        dataset=dataset,
        data_processor=data_processor,
        index=index,
        device=device,
    )
    errors = _relative_errors(pred, truth, d=2)

    vmin = min(pred.min().item(), truth.min().item())
    vmax = max(pred.max().item(), truth.max().item())

    n_panels = 4 if plot_k else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    solution_start = 0
    if plot_k:
        coefficient_image = axes[0].imshow(coefficient, origin="lower")
        axes[0].set_title("Coefficient k")
        fig.colorbar(coefficient_image, ax=axes[0], shrink=0.75)
        solution_start = 1

    solution_image = axes[solution_start].imshow(
        truth, origin="lower", vmin=vmin, vmax=vmax
    )
    axes[solution_start].set_title("Ground truth")
    axes[solution_start + 1].imshow(pred, origin="lower", vmin=vmin, vmax=vmax)
    axes[solution_start + 1].set_title("Prediction")
    error_image = axes[solution_start + 2].imshow(
        (pred - truth).abs(), origin="lower"
    )
    axes[solution_start + 2].set_title("Absolute error")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(
        solution_image,
        ax=axes[solution_start : solution_start + 2],
        shrink=0.75,
    )
    fig.colorbar(error_image, ax=axes[solution_start + 2], shrink=0.75)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    if print_errors:
        print(f"Plotted sample {index}: relative H1={errors['h1']:.4f}, relative L2={errors['l2']:.4f}")
    plt.show()

    return fig, axes, errors



device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("neuralop/data/datasets/poisson1d_dataset.pt")
ckpt_dir = Path("./ckpt/colab_1d_darcy")

train_loader, test_loaders, data_processor = load_multiop_darcy(
    data_path=data_path,
    n_train_operators=100000,
    n_test_operators=20000,
    n_context=8,
    batch_size=32,
    test_batch_size=32,
    n_train_samples=20000,
    n_test_samples=2000,
    encode_input=True,
    encode_output=True,
    operator_direction="f_to_u",
)

data_processor = data_processor.to(device)

# loading test dataset on unseen operators
raw_dataset = train_loader.dataset
final_test_dataset = MultiOperatorDarcyDataset(
    k=raw_dataset.k,
    f=raw_dataset.f,
    u=raw_dataset.u,
    operator_indices=torch.arange(120000, 125000),
    n_context=8,
    n_samples=5000,
    random_context=False,
    operator_direction="f_to_u",
)

final_test_loader = DataLoader(
    final_test_dataset,
    batch_size=32,
    shuffle=False,
)

model = FNOSets(
    n_modes=(16,),
    in_channels=1,
    out_channels=1,
    hidden_channels=256,
    encoder_layers=4,
    decoder_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    enforce_hermitian_symmetry=False,
)

model, _, _, _, epoch = load_training_state(
    save_dir=ckpt_dir,
    save_name="best_model",
    model=model,
    map_location=device,
)

model = model.to(device)
model.eval()

l2loss = LpLoss(d=1, p=2)
h1loss = H1Loss(d=1)
eval_losses = {"h1": h1loss, "l2": l2loss}

trainer = Trainer(
    model=model,
    n_epochs=1,
    device=device,
    data_processor=data_processor,
    verbose=True,
)

metrics = trainer.evaluate(
    loss_dict=eval_losses,
    data_loader=final_test_loader,
    log_prefix="test",
)

print(metrics)

for i in range(20):
    plot_fnosets_prediction_1d(
        model=model,
        dataset=final_test_loader.dataset,
        data_processor=data_processor,
        index=random.randrange(len(final_test_loader.dataset)),
        device=device,
        save_path=f"fnosets_prediction_1d_{i}.png",
    )
