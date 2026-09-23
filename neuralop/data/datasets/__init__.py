from .darcy import DarcyDataset, load_darcy_flow_small
from .multioperator_darcy import (
    MultiOperatorDarcyDataProcessor,
    MultiOperatorDarcyDataset,
    load_multiop_darcy,
)
from .chunked_multioperator_darcy import (
    ChunkedMultiOperatorDarcyDataset,
    load_chunked_multiop_darcy,
)
from .fnosets_inverse import (
    SCALAR_COEFFICIENT_REPRESENTATION,
    SYMMETRIC_2D_COEFFICIENT_REPRESENTATION,
    SYMMETRIC_2D_COMPONENTS,
    CoefficientTargetDataset,
    FNOSetsInverseDataProcessor,
    RelativeFrobeniusLoss,
    build_fnosets_inverse_data_processor,
    channels_to_symmetric_tensor,
    fit_coefficient_normalizer,
    load_fnosets_inverse_data_processor,
    relative_frobenius_error,
    save_fnosets_inverse_data_processor,
    smallest_symmetric_eigenvalue,
    symmetric_tensor_to_channels,
    wrap_fnosets_inverse_loader,
    wrap_fnosets_inverse_loaders,
)
from .navier_stokes import NavierStokesDataset, load_navier_stokes_pt
from .pt_dataset import PTDataset
from .burgers import Burgers1dTimeDataset, load_mini_burgers_1dtime
from .dict_dataset import DictDataset
from .mesh_datamodule import MeshDataModule
from .car_cfd_dataset import CarCFDDataset, load_mini_car
from .ot_datamodule import OTDataModule
from .car_ot_dataset import CarOTDataset, load_saved_ot, CFDDataProcessor

# only import SphericalSWEDataset if torch_harmonics is built locally
try:
    from .spherical_swe import load_spherical_swe
except ModuleNotFoundError:
    pass

# only import TheWell if the_well is built
try:
    from .the_well_dataset import TheWellDataset, ActiveMatterDataset, MHD64Dataset
except ModuleNotFoundError:
    pass
