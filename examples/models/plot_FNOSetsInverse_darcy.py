"""
Training FNOSetsInverse on Multi-Operator Darcy/Poisson Data
===========================================================

We train an FNOSetsInverse model to infer the coefficient field ``k`` from
episodic in-context forcing/solution pairs.
"""

from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from neuralop.models.fnosets import FNOSetsInverse
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_multiop_darcy
from neuralop.data.transforms.data_processors import DataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss


class CoefficientTargetDataset(Dataset):
    """Wrap a Darcy episodic dataset so the training target is ``k``."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["y"] = sample["k"]
        return sample


class FNOSetsInverseDataProcessor(DataProcessor):
    """Normalize context fields and the coefficient target."""

    def __init__(self, input_normalizer=None, output_normalizer=None, coefficient_normalizer=None):
        super().__init__()
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.coefficient_normalizer = coefficient_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.input_normalizer is not None:
            self.input_normalizer = self.input_normalizer.to(device)
        if self.output_normalizer is not None:
            self.output_normalizer = self.output_normalizer.to(device)
        if self.coefficient_normalizer is not None:
            self.coefficient_normalizer = self.coefficient_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                data_dict[key] = value.to(self.device)

        if self.input_normalizer is not None:
            data_dict["u_context"] = self.input_normalizer.transform(data_dict["u_context"])
        if self.output_normalizer is not None:
            data_dict["f_context"] = self.output_normalizer.transform(data_dict["f_context"])
        if self.coefficient_normalizer is not None and self.training:
            data_dict["y"] = self.coefficient_normalizer.transform(data_dict["y"])

        return data_dict

    def postprocess(self, output, data_dict):
        if self.coefficient_normalizer is not None and not self.training:
            output = self.coefficient_normalizer.inverse_transform(output)
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict


def fit_coefficient_normalizer(dataset):
    operator_indices = dataset.operator_indices
    train_k = dataset.k[operator_indices].unsqueeze(1).float()
    reduce_dims = list(range(train_k.ndim))
    reduce_dims.pop(1)
    normalizer = UnitGaussianNormalizer(dim=reduce_dims)
    normalizer.fit(train_k)
    return normalizer


def wrap_loader(loader, shuffle):
    return DataLoader(
        CoefficientTargetDataset(loader.dataset),
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
    )


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

    pred = pred[0, 0].detach().cpu()
    truth = sample["y"][0, 0].detach().cpu()
    vmin = min(pred.min().item(), truth.min().item())
    vmax = max(pred.max().item(), truth.max().item())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im = axes[0].imshow(truth, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth k")
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
    plt.show()
    return fig, axes


device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("neuralop/data/datasets/poisson1d_dataset.pt")
spatial_dim = 1

train_loader, test_loaders, base_data_processor = load_multiop_darcy(
    data_path=data_path,
    n_train_operators=1000,
    n_test_operators=200,
    n_context=8,
    batch_size=16,
    test_batch_size=16,
    n_train_samples=256,
    n_test_samples=64,
    encode_input=True,
    encode_output=True,
    include_k=True,
    operator_direction="f_to_u",
)

data_processor = FNOSetsInverseDataProcessor(
    input_normalizer=base_data_processor.in_normalizer,
    output_normalizer=base_data_processor.out_normalizer,
    coefficient_normalizer=fit_coefficient_normalizer(train_loader.dataset),
).to(device)

train_loader = wrap_loader(train_loader, shuffle=True)
test_loaders = {
    name: wrap_loader(loader, shuffle=False)
    for name, loader in test_loaders.items()
}

model = FNOSetsInverse(
    n_modes=(16,),
    in_channels=1,
    out_channels=1,
    coefficient_channels=1,
    hidden_channels=256,
    encoder_layers=4,
    decoder_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    enforce_hermitian_symmetry=False,
)
model = model.to(device)

n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=spatial_dim, p=2)
h1loss = H1Loss(d=spatial_dim)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=10,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=5,
    use_distributed=False,
    verbose=True,
    progress_bar=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_best="test_h1",
    save_dir="./ckpt/fnosets_inverse_1d_darcy",
)
