"""NeurInferno: field-boundary inference for unlabeled binary protocols."""

from importlib.metadata import PackageNotFoundError, version

from neurinferno.data_generation.label_format import FIELD_TYPE_IDS, FIELD_TYPES
from neurinferno.inference import (
    DEFAULT_REPO,
    FieldBoundaryModel,
    MessageResult,
    Segment,
    infer_boundaries,
    parse_hex,
)
from neurinferno.model.full_model import FullModel

__all__ = [
    "DEFAULT_REPO",
    "FIELD_TYPES",
    "FIELD_TYPE_IDS",
    "FieldBoundaryModel",
    "FullModel",
    "MessageResult",
    "Segment",
    "infer_boundaries",
    "parse_hex",
]

try:
    __version__ = version("neurinferno")
except PackageNotFoundError:
    __version__ = "0.2.0"
