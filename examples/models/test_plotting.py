from pathlib import Path

import torch
import random
import matplotlib.pyplot as plt

from neuralop.models import FNOSets
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training.training_state import load_training_state
from neuralop.data.datasets import load_multiop_darcy

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
):
    """Plot 2D FNOSets prediction and ground truth for one episodic sample."""
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im = axes[0].imshow(truth, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth")
    axes[1].imshow(pred, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[2].imshow((pred - truth).abs(), origin="lower")
    axes[2].set_title("Absolute error")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes[:2], shrink=0.75)
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
    data_loader=test_loaders["test"],
    log_prefix="test",
)

print(metrics)

plot_fnosets_prediction_1d(
    model=model,
    dataset=test_loaders["test"].dataset,
    data_processor=data_processor,
    index=random.randrange(len(test_loaders["test"].dataset)),
    device=device,
    save_path="fnosets_prediction_1d.png",
)
