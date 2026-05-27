from pathlib import Path
import random

import torch

from neuralop.models.fnosets import FNOSetsInverse
from neuralop import LpLoss, H1Loss
from neuralop.data.datasets import load_multiop_darcy
from neuralop.data.transforms.data_processors import DataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.training.training_state import load_training_state


class FNOSetsInverseDataProcessor(DataProcessor):
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


device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("neuralop/data/datasets/poisson1d_dataset.pt")
ckpt_dir = Path("./ckpt/fnosets_inverse_1d_darcy")
output_path = Path("fnosets_inverse_predictions.pt")

n_train_operators = 1000
n_test_operators = 200
n_context = 10
n_samples = 20
pair_index_to_save = 10
spatial_dim = 1

train_loader, _, base_data_processor = load_multiop_darcy(
    data_path=data_path,
    n_train_operators=n_train_operators,
    n_test_operators=n_test_operators,
    n_context=n_context,
    batch_size=1,
    test_batch_size=1,
    n_train_samples=1,
    n_test_samples=1,
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
data_processor.eval()

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

model, _, _, _, epoch = load_training_state(
    save_dir=ckpt_dir,
    save_name="best_model",
    model=model,
    map_location=device,
)
model = model.to(device)
model.eval()

k_data = train_loader.dataset.k
f_data = train_loader.dataset.f
u_data = train_loader.dataset.u

test_operator_start = n_train_operators
test_operator_stop = n_train_operators + n_test_operators
candidate_operator_indices = list(range(test_operator_start, test_operator_stop))
if len(candidate_operator_indices) < n_samples:
    raise ValueError(
        f"Need at least {n_samples} test operators, got {len(candidate_operator_indices)}."
    )
operator_indices = random.sample(candidate_operator_indices, n_samples)

if f_data.shape[1] <= pair_index_to_save or u_data.shape[1] <= pair_index_to_save:
    raise ValueError(
        f"Need at least {pair_index_to_save + 1} pairs per operator to save the 11th pair, "
        f"got f.shape[1]={f_data.shape[1]} and u.shape[1]={u_data.shape[1]}."
    )

spatial_shape = k_data.shape[1:]
k_predictions = torch.empty(n_samples, *spatial_shape)
k_true = torch.empty(n_samples, *spatial_shape)
u_true = torch.empty(n_samples, *spatial_shape)
f_true = torch.empty(n_samples, *spatial_shape)

l2loss = LpLoss(d=spatial_dim, p=2)
h1loss = H1Loss(d=spatial_dim)

for i, operator_idx in enumerate(operator_indices):
    sample = {
        "u_context": f_data[operator_idx, :n_context].unsqueeze(1).unsqueeze(0),
        "f_context": u_data[operator_idx, :n_context].unsqueeze(1).unsqueeze(0),
        "y": k_data[operator_idx].unsqueeze(0).unsqueeze(0),
    }

    sample = data_processor.preprocess(sample)

    with torch.no_grad():
        pred = model(**sample)
        pred, sample = data_processor.postprocess(pred, sample)

    pred_k = pred[0, 0].detach().cpu()
    true_k = k_data[operator_idx].detach().cpu()

    k_predictions[i] = pred_k
    k_true[i] = true_k
    u_true[i] = u_data[operator_idx, pair_index_to_save].detach().cpu()
    f_true[i] = f_data[operator_idx, pair_index_to_save].detach().cpu()

    pred_batch = pred_k.unsqueeze(0).unsqueeze(0)
    true_batch = true_k.unsqueeze(0).unsqueeze(0)
    l2_error = l2loss(pred_batch, true_batch).item()
    h1_error = h1loss(pred_batch, true_batch).item()

    print(
        f"Sample {i:02d}, operator {operator_idx}: "
        f"relative H1={h1_error:.6f}, relative L2={l2_error:.6f}"
    )

torch.save(
    {
        "operator_indices": torch.tensor(operator_indices, dtype=torch.long),
        "k_predictions": k_predictions,
        "k_true": k_true,
        "u_true": u_true,
        "f_true": f_true,
    },
    output_path,
)

print(f"Saved tensors to {output_path}")
print(f"k_predictions shape: {tuple(k_predictions.shape)}")
print(f"k_true shape: {tuple(k_true.shape)}")
print(f"u_true shape: {tuple(u_true.shape)}")
print(f"f_true shape: {tuple(f_true.shape)}")
