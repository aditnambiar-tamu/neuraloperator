import pytest
import torch

from neuralop.models import FNOSetsCalderon


def _model(output_shape=None):
    return FNOSetsCalderon(
        boundary_n_modes=(4,),
        coefficient_n_modes=(4, 4),
        output_shape=output_shape,
        hidden_channels=8,
        encoder_layers=1,
        decoder_layers=1,
        channel_mlp_expansion=1.0,
        enforce_hermitian_symmetry=True,
    )


def test_fnosets_calderon_maps_1d_contexts_to_2d_coefficient_and_backpropagates():
    model = _model(output_shape=(7, 9))
    f_boundary = torch.randn(2, 3, 1, 16)
    n_boundary = torch.randn(2, 3, 1, 16)
    target = torch.randn(2, 1, 7, 9)

    output = model(f_boundary, n_boundary, y=target)

    assert output.shape == target.shape
    output.square().mean().backward()
    assert model.f_boundary_encoder.lifting.fcs[0].weight.grad is not None
    assert model.n_boundary_encoder.lifting.fcs[0].weight.grad is not None
    assert model.decoder.projection.fcs[-1].weight.grad is not None


def test_fnosets_calderon_uses_target_shape_when_no_shape_is_configured():
    model = _model()
    output = model(
        torch.randn(1, 2, 1, 16),
        torch.randn(1, 2, 1, 16),
        y=torch.randn(1, 1, 6, 10),
    )
    assert output.shape == (1, 1, 6, 10)


def test_fnosets_calderon_predicts_without_target_when_shape_is_configured():
    model = _model(output_shape=(5, 8))
    output = model(torch.randn(2, 1, 1, 16), torch.randn(2, 1, 1, 16))
    assert output.shape == (2, 1, 5, 8)


def test_fnosets_calderon_checks_boundary_pair_shapes():
    model = _model(output_shape=(7, 9))
    with pytest.raises(ValueError, match="boundary-point count"):
        model(torch.randn(2, 3, 1, 16), torch.randn(2, 3, 1, 15))


def test_fnosets_calderon_checks_target_shape():
    model = _model(output_shape=(7, 9))
    with pytest.raises(ValueError, match="output and target shapes"):
        model(
            torch.randn(2, 3, 1, 16),
            torch.randn(2, 3, 1, 16),
            y=torch.randn(2, 1, 8, 9),
        )
