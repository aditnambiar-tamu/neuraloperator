from functools import partialmethod
from typing import Tuple, List, Union, Literal, Optional

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)


from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel

class FNOEncoder(nn.Module):
    """FNO-style encoder that maps an input function to a latent function.

    This mirrors the front half of :class:`FNO`: optional positional embedding,
    lifting, optional domain padding, and a stack of FNO blocks. It intentionally
    does not include a projection layer because FNOSets will consume the latent
    representation directly.
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        norm_groups: int = 1,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()

        self.n_dim = len(n_modes)
        self._n_modes = n_modes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * hidden_channels)
        self.non_linearity = non_linearity
        self.complex_data = complex_data

        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNOEncoder positional embedding with {positional_embedding}, "
                "expected one of 'grid', GridEmbeddingND, GridEmbedding2D, or None"
            )

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            norm_groups=norm_groups,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )

        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        self.lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

    def forward(self, x, output_shape=None, **kwargs):
        """Encode ``x`` into a hidden-channel latent function."""
        if kwargs:
            warnings.warn(
                f"FNOEncoder.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

class FNODecoder(nn.Module):
    """FNO-style decoder that maps a latent function to an output function."""

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 4,
        in_channels: int = None,
        projection_channel_ratio: Number = 2,
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        norm_groups: int = 1,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()

        self.n_dim = len(n_modes)
        self._n_modes = n_modes
        self.in_channels = in_channels if in_channels is not None else 2 * hidden_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * hidden_channels)
        self.non_linearity = non_linearity
        self.complex_data = complex_data

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            norm_groups=norm_groups,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )

        self.context_query_mixer = ChannelMLP(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.context_query_mixer = ComplexValued(self.context_query_mixer)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, **kwargs):
        """Decode latent representation ``x`` into output-channel function values."""
        if kwargs:
            warnings.warn(
                f"FNODecoder.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        x = self.context_query_mixer(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

class FNOSets(BaseModel, name='FNOSets'):
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        norm_groups: int = 1,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        encoder_resolution_scaling_factor: Union[Number, List[Number]] = None,
        decoder_resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()

        self.n_dim = len(n_modes)
        self._n_modes = n_modes

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.complex_data = complex_data

        encoder_kwargs = dict(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=encoder_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            norm_groups=norm_groups,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=encoder_resolution_scaling_factor,
            domain_padding=domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )
        self.u_context_encoder = FNOEncoder(
            in_channels=in_channels,
            **encoder_kwargs,
        )
        self.f_context_encoder = FNOEncoder(
            in_channels=out_channels,
            **encoder_kwargs,
        )
        self.query_encoder = FNOEncoder(
            in_channels=in_channels,
            **encoder_kwargs,
        )

        pair_mixer_channels = int(projection_channel_ratio * hidden_channels)
        self.context_pair_mixer = ChannelMLP(
            in_channels=2 * hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=pair_mixer_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.context_pair_mixer = ComplexValued(self.context_pair_mixer)

        self.decoder = FNODecoder(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_layers=decoder_layers,
            projection_channel_ratio=projection_channel_ratio,
            non_linearity=non_linearity,
            norm=norm,
            norm_groups=norm_groups,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=decoder_resolution_scaling_factor,
            domain_padding=domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )

    def forward(
        self,
        u_context,
        f_context,
        u_query,
        encoder_output_shape=None,
        decoder_output_shape=None,
        **kwargs,
    ):
        """Predict the query output from in-context examples and a query input.

        Parameters
        ----------
        u_context : torch.Tensor
            Tensor of input-side context functions with shape
            ``(batch, n_context, in_channels, *spatial_shape)``.
        f_context : torch.Tensor
            Tensor of output-side context functions with shape
            ``(batch, n_context, out_channels, *spatial_shape)``.
        u_query : torch.Tensor
            Query input function with shape ``(batch, in_channels, *spatial_shape)``.
        """
        if kwargs:
            warnings.warn(
                f"FNOSets.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if u_context.ndim != f_context.ndim:
            raise ValueError(
                "Expected u_context and f_context to have the same number of dimensions, "
                f"got {u_context.ndim} and {f_context.ndim}."
            )
        if u_context.shape[0] != f_context.shape[0] or u_context.shape[1] != f_context.shape[1]:
            raise ValueError(
                "Expected u_context and f_context to agree on batch and context dimensions, "
                f"got {u_context.shape[:2]} and {f_context.shape[:2]}."
            )
        if u_context.shape[0] != u_query.shape[0]:
            raise ValueError(
                "Expected u_context and u_query to agree on batch dimension, "
                f"got {u_context.shape[0]} and {u_query.shape[0]}."
            )

        batch_size, n_context = u_context.shape[:2]
        u_context_shape = u_context.shape[2:]
        f_context_shape = f_context.shape[2:]

        u_context = u_context.reshape(batch_size * n_context, *u_context_shape)
        f_context = f_context.reshape(batch_size * n_context, *f_context_shape)

        h_u = self.u_context_encoder(u_context, output_shape=encoder_output_shape)
        h_f = self.f_context_encoder(f_context, output_shape=encoder_output_shape)

        h_context = torch.cat([h_u, h_f], dim=1)
        h_context = self.context_pair_mixer(h_context) # MLP to turn (u,f) pair into h

        h_context = h_context.reshape(
            batch_size,
            n_context,
            self.hidden_channels,
            *h_context.shape[2:],
        )
        h_agg = h_context.mean(dim=1)

        h_q = self.query_encoder(u_query, output_shape=encoder_output_shape)
        decoder_input = torch.cat([h_agg, h_q], dim=1)

        return self.decoder(decoder_input, output_shape=decoder_output_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.u_context_encoder.n_modes = n_modes
        self.f_context_encoder.n_modes = n_modes
        self.query_encoder.n_modes = n_modes
        self.decoder.n_modes = n_modes
        self._n_modes = n_modes


class FNOSetsInverse(BaseModel, name='FNOSetsInverse'):
    """Infer a coefficient field from in-context input-output function pairs.

    The model encodes each context pair independently, mixes the paired latent
    representation, mean-pools across the context set, and decodes the pooled
    latent field into the coefficient function. Use ``coefficient_channels=1``
    for scalar coefficients or ``coefficient_channels=3`` with the inverse
    data processor's symmetric 2D ``(K11, K22, K12)`` representation.
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        coefficient_channels: int,
        hidden_channels: int,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        norm_groups: int = 1,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        encoder_resolution_scaling_factor: Union[Number, List[Number]] = None,
        decoder_resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()

        self.n_dim = len(n_modes)
        self._n_modes = n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coefficient_channels = coefficient_channels
        self.hidden_channels = hidden_channels
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.complex_data = complex_data

        encoder_kwargs = dict(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=encoder_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            norm_groups=norm_groups,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=encoder_resolution_scaling_factor,
            domain_padding=domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )
        self.u_context_encoder = FNOEncoder(
            in_channels=in_channels,
            **encoder_kwargs,
        )
        self.f_context_encoder = FNOEncoder(
            in_channels=out_channels,
            **encoder_kwargs,
        )

        pair_mixer_channels = int(projection_channel_ratio * hidden_channels)
        self.context_pair_mixer = ChannelMLP(
            in_channels=2 * hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=pair_mixer_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.context_pair_mixer = ComplexValued(self.context_pair_mixer)

        self.decoder = FNODecoder(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            out_channels=coefficient_channels,
            in_channels=hidden_channels,
            n_layers=decoder_layers,
            projection_channel_ratio=projection_channel_ratio,
            non_linearity=non_linearity,
            norm=norm,
            norm_groups=norm_groups,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=decoder_resolution_scaling_factor,
            domain_padding=domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )

    def forward(
        self,
        u_context,
        f_context,
        u_query=None,
        y=None,
        k=None,
        encoder_output_shape=None,
        decoder_output_shape=None,
        **kwargs,
    ):
        """Predict a coefficient field from context function pairs.

        ``u_context`` and ``f_context`` follow the same key names as
        ``FNOSets``. With ``operator_direction="f_to_u"`` in the Darcy loader,
        ``u_context`` contains the forcing functions and ``f_context`` contains
        the solution functions.
        """
        if kwargs:
            warnings.warn(
                f"FNOSetsInverse.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if u_context.ndim != f_context.ndim:
            raise ValueError(
                "Expected u_context and f_context to have the same number of dimensions, "
                f"got {u_context.ndim} and {f_context.ndim}."
            )
        if u_context.shape[0] != f_context.shape[0] or u_context.shape[1] != f_context.shape[1]:
            raise ValueError(
                "Expected u_context and f_context to agree on batch and context dimensions, "
                f"got {u_context.shape[:2]} and {f_context.shape[:2]}."
            )
        if y is not None and (
            y.ndim < 2 or y.shape[1] != self.coefficient_channels
        ):
            target_channels = y.shape[1] if y.ndim >= 2 else None
            raise ValueError(
                "FNOSetsInverse target channels must match coefficient_channels; "
                f"model expects {self.coefficient_channels}, target has "
                f"{target_channels}."
            )

        batch_size, n_context = u_context.shape[:2]
        u_context_shape = u_context.shape[2:]
        f_context_shape = f_context.shape[2:]

        u_context = u_context.reshape(batch_size * n_context, *u_context_shape)
        f_context = f_context.reshape(batch_size * n_context, *f_context_shape)

        h_u = self.u_context_encoder(u_context, output_shape=encoder_output_shape)
        h_f = self.f_context_encoder(f_context, output_shape=encoder_output_shape)

        h_context = torch.cat([h_u, h_f], dim=1)
        h_context = self.context_pair_mixer(h_context)

        h_context = h_context.reshape(
            batch_size,
            n_context,
            self.hidden_channels,
            *h_context.shape[2:],
        )
        h_agg = h_context.mean(dim=1)

        output = self.decoder(h_agg, output_shape=decoder_output_shape)
        if y is not None and output.shape != y.shape:
            raise ValueError(
                "FNOSetsInverse output and target shapes must match; got "
                f"output={tuple(output.shape)} and target={tuple(y.shape)}."
            )
        return output

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.u_context_encoder.n_modes = n_modes
        self.f_context_encoder.n_modes = n_modes
        self.decoder.n_modes = n_modes
        self._n_modes = n_modes


class FNOSetsCalderon(BaseModel, name="FNOSetsCalderon"):
    """Recover a 2D coefficient field from paired 1D boundary traces.

    Each context example contains one Dirichlet trace and its corresponding
    Neumann trace. Separate 1D FNO encoders process the two traces, a pointwise
    mixer combines each pair, and mean pooling over boundary locations and
    context examples produces an operator-level representation. That vector is
    broadcast over the coefficient grid, augmented with 2D grid coordinates,
    and decoded into the coefficient field.

    Boundary points are treated as an ordered, uniformly sampled 1D sequence.
    The same point ordering must be used for every example and operator.
    """

    def __init__(
        self,
        boundary_n_modes: Tuple[int, ...] = (16,),
        coefficient_n_modes: Tuple[int, ...] = (16, 16),
        output_shape: Optional[Tuple[int, int]] = None,
        f_boundary_channels: int = 1,
        n_boundary_channels: int = 1,
        coefficient_channels: int = 1,
        hidden_channels: int = 128,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        norm_groups: int = 1,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal[
            "linear", "identity", "soft-gating", None
        ] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        boundary_domain_padding: Union[Number, List[Number]] = None,
        coefficient_domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_boundary_n_modes: Tuple[int, ...] = None,
        max_coefficient_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()

        if isinstance(boundary_n_modes, int):
            boundary_n_modes = (boundary_n_modes,)
        if isinstance(coefficient_n_modes, int):
            coefficient_n_modes = (coefficient_n_modes, coefficient_n_modes)
        if len(boundary_n_modes) != 1:
            raise ValueError(
                "boundary_n_modes must specify exactly one dimension for the "
                f"1D boundary traces, got {boundary_n_modes}."
            )
        if len(coefficient_n_modes) != 2:
            raise ValueError(
                "coefficient_n_modes must specify two dimensions for the "
                f"2D coefficient grid, got {coefficient_n_modes}."
            )
        if output_shape is not None:
            output_shape = tuple(int(size) for size in output_shape)
            if len(output_shape) != 2 or any(size < 1 for size in output_shape):
                raise ValueError(
                    "output_shape must contain two positive spatial sizes, "
                    f"got {output_shape}."
                )
        if min(f_boundary_channels, n_boundary_channels, coefficient_channels) < 1:
            raise ValueError("All input and output channel counts must be positive.")
        if hidden_channels < 1 or encoder_layers < 1 or decoder_layers < 1:
            raise ValueError(
                "hidden_channels, encoder_layers, and decoder_layers must be positive."
            )

        self._boundary_n_modes = tuple(boundary_n_modes)
        self._coefficient_n_modes = tuple(coefficient_n_modes)
        self.output_shape = output_shape
        self.f_boundary_channels = int(f_boundary_channels)
        self.n_boundary_channels = int(n_boundary_channels)
        self.coefficient_channels = int(coefficient_channels)
        self.hidden_channels = int(hidden_channels)

        encoder_kwargs = dict(
            hidden_channels=hidden_channels,
            n_layers=encoder_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            norm_groups=norm_groups,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            domain_padding=boundary_domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_boundary_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )
        self.f_boundary_encoder = FNOEncoder(
            n_modes=self._boundary_n_modes,
            in_channels=self.f_boundary_channels,
            **encoder_kwargs,
        )
        self.n_boundary_encoder = FNOEncoder(
            n_modes=self._boundary_n_modes,
            in_channels=self.n_boundary_channels,
            **encoder_kwargs,
        )

        pair_mixer_channels = int(projection_channel_ratio * hidden_channels)
        self.context_pair_mixer = ChannelMLP(
            in_channels=2 * hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=pair_mixer_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=non_linearity,
        )
        self.coefficient_grid_embedding = GridEmbeddingND(
            in_channels=hidden_channels,
            dim=2,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )

        self.decoder = FNODecoder(
            n_modes=self._coefficient_n_modes,
            hidden_channels=hidden_channels,
            out_channels=self.coefficient_channels,
            in_channels=hidden_channels + 2,
            n_layers=decoder_layers,
            projection_channel_ratio=projection_channel_ratio,
            non_linearity=non_linearity,
            norm=norm,
            norm_groups=norm_groups,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            domain_padding=coefficient_domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_coefficient_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )

    def forward(
        self,
        f_boundary_context,
        n_boundary_context,
        y=None,
        output_shape=None,
        **kwargs,
    ):
        """Predict a 2D coefficient field from paired boundary contexts.

        Parameters
        ----------
        f_boundary_context : torch.Tensor
            Shape ``(batch, n_context, channels, n_boundary_points)``.
        n_boundary_context : torch.Tensor
            Paired Neumann traces with the same batch, context, and point axes.
        y : torch.Tensor, optional
            Coefficient target. Its spatial shape is used when neither the
            forward call nor constructor specifies ``output_shape``; values are
            only used for shape validation, never to form the prediction.
        output_shape : tuple, optional
            Requested coefficient-grid shape, as ``(height, width)``. Defaults
            to the constructor's ``output_shape`` and then to ``y``'s shape.
        """

        if kwargs:
            warnings.warn(
                "FNOSetsCalderon.forward() received unexpected keyword "
                f"arguments: {list(kwargs.keys())}. These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if f_boundary_context.ndim != 4 or n_boundary_context.ndim != 4:
            raise ValueError(
                "Expected boundary contexts to have shape "
                "(batch, n_context, channels, n_boundary_points); got "
                f"{tuple(f_boundary_context.shape)} and "
                f"{tuple(n_boundary_context.shape)}."
            )
        if any(size < 1 for size in f_boundary_context.shape):
            raise ValueError("Boundary contexts must not contain empty dimensions.")
        if (
            f_boundary_context.shape[:2] != n_boundary_context.shape[:2]
            or f_boundary_context.shape[-1] != n_boundary_context.shape[-1]
        ):
            raise ValueError(
                "Dirichlet and Neumann contexts must agree on batch size, "
                "context count, and boundary-point count."
            )
        if f_boundary_context.shape[2] != self.f_boundary_channels:
            raise ValueError(
                f"Expected {self.f_boundary_channels} Dirichlet channels, got "
                f"{f_boundary_context.shape[2]}."
            )
        if n_boundary_context.shape[2] != self.n_boundary_channels:
            raise ValueError(
                f"Expected {self.n_boundary_channels} Neumann channels, got "
                f"{n_boundary_context.shape[2]}."
            )

        if output_shape is None:
            output_shape = self.output_shape
        if output_shape is None and y is not None:
            output_shape = tuple(y.shape[-2:])
        if output_shape is None:
            raise ValueError(
                "Specify output_shape in the constructor or forward call when "
                "predicting without a coefficient target."
            )
        output_shape = tuple(int(size) for size in output_shape)
        if len(output_shape) != 2 or any(size < 1 for size in output_shape):
            raise ValueError(
                f"output_shape must contain two positive sizes, got {output_shape}."
            )

        batch_size, n_context = f_boundary_context.shape[:2]
        f_shape = f_boundary_context.shape[2:]
        n_shape = n_boundary_context.shape[2:]
        f_flat = f_boundary_context.reshape(batch_size * n_context, *f_shape)
        n_flat = n_boundary_context.reshape(batch_size * n_context, *n_shape)
        h_f = self.f_boundary_encoder(f_flat)
        h_n = self.n_boundary_encoder(n_flat)
        pair_features = self.context_pair_mixer(torch.cat([h_f, h_n], dim=1))
        pair_features = pair_features.reshape(
            batch_size,
            n_context,
            self.hidden_channels,
            pair_features.shape[-1],
        )
        operator_features = pair_features.mean(dim=-1).mean(dim=1)
        coefficient_latent = operator_features[:, :, None, None].expand(
            -1, -1, output_shape[0], output_shape[1]
        )
        decoder_input = self.coefficient_grid_embedding(coefficient_latent)
        output = self.decoder(decoder_input)

        if y is not None and output.shape != y.shape:
            raise ValueError(
                "FNOSetsCalderon output and target shapes must match; got "
                f"output={tuple(output.shape)} and target={tuple(y.shape)}."
            )
        return output

    @property
    def boundary_n_modes(self):
        return self._boundary_n_modes

    @boundary_n_modes.setter
    def boundary_n_modes(self, modes):
        if isinstance(modes, int):
            modes = (modes,)
        if len(modes) != 1:
            raise ValueError("boundary_n_modes must contain exactly one value.")
        self.f_boundary_encoder.n_modes = modes
        self.n_boundary_encoder.n_modes = modes
        self._boundary_n_modes = tuple(modes)

    @property
    def coefficient_n_modes(self):
        return self._coefficient_n_modes

    @coefficient_n_modes.setter
    def coefficient_n_modes(self, modes):
        if isinstance(modes, int):
            modes = (modes, modes)
        if len(modes) != 2:
            raise ValueError("coefficient_n_modes must contain exactly two values.")
        self.decoder.n_modes = modes
        self._coefficient_n_modes = tuple(modes)
