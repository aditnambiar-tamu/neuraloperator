"""
Training FNOSets on Multi-Operator 1D Darcy/Poisson Data
========================================================

We train an FNOSets model on an episodic multi-operator 1D dataset. Each
training sample contains several in-context input-output pairs from one
operator and one query input whose output should be predicted.

This example demonstrates the core workflow:
1. Loading and preprocessing the episodic 1D dataset
2. Creating an FNOSets model architecture
3. Setting up training components (optimizer, scheduler, losses)
4. Training the model
"""

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Import dependencies
# -------------------
# We import the necessary modules from `neuralop` for training FNOSets.

from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

from neuralop.models import FNOSets
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_multiop_darcy
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = "cpu"

# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Loading the Multi-Operator 1D Dataset
# -------------------------------------
# The dataset file stores operator-indexed arrays with keys ``k``, ``f``, and
# ``u``. The loader turns these arrays into episodic batches compatible with
# ``FNOSets`` and the standard ``Trainer``.

data_path = Path("neuralop/data/datasets/poisson1d_dataset.pt")

train_loader, test_loaders, data_processor = load_multiop_darcy(
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
)
data_processor = data_processor.to(device)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the FNOSets Model
# --------------------------
# FNOSets encodes in-context pairs, mean-pools their embeddings, combines the
# pooled context with the encoded query function, and decodes the query output.

model = FNOSets(
    # For 1D data, use n_modes=(m,). For 2D data, use n_modes=(m_x, m_y),
    # for example n_modes=(16, 16).
    n_modes=(16,),
    in_channels=1,
    out_channels=1,
    hidden_channels=256,
    encoder_layers=4,
    decoder_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    channel_mlp_expansion=2.0,
    # This is needed for the current 1D real-valued FFT path. For 2D, the
    # default enforce_hermitian_symmetry=True is usually appropriate.
    enforce_hermitian_symmetry=False,
)
model = model.to(device)

# Count and display the number of parameters
n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the Optimizer and Scheduler
# ------------------------------------
# We use AdamW optimizer with weight decay for regularization.

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Setting up Loss Functions
# -------------------------
# We use H1 loss for training and L2 loss for evaluation.

# For 1D data, use d=1. For 2D data, use d=2.
l2loss = LpLoss(d=1, p=2)
h1loss = H1Loss(d=1)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training the Model
# ------------------
# We display the training configuration and then train the model.

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Creating the Trainer
# --------------------
# The standard Trainer can be reused because each batch is a dictionary whose
# keys match the FNOSets forward signature plus the target key ``y``.

trainer = Trainer(
    model=model,
    n_epochs=10,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=5,
    use_distributed=False,
    verbose=True,
    progress_bar=True
)


# %%
# .. raw:: html
#
#    <div style="margin-top: 3em;"></div>
#
# Training
# --------
# The trainer will:
# 1. Run the forward pass through FNOSets
# 2. Compute the H1 loss against the query target
# 3. Backpropagate and update weights
# 4. Evaluate every ``eval_interval`` epochs

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_best="test_h1",
    save_dir="./ckpt/fnosets_1d_darcy",
)