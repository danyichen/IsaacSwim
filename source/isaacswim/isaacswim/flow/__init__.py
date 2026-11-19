"""Current fields: analytic flows, gridded ocean data and CFD."""

from isaacswim.flow.fields import (
    BoundaryLayer,
    Composite,
    GustyCurrent,
    Jet,
    Pipe,
    Uniform,
    VelocityField,
    shear_profile,
)
from isaacswim.flow.gridded import Gridded

__all__ = [
    "VelocityField",
    "Uniform",
    "GustyCurrent",
    "Jet",
    "Pipe",
    "Gridded",
    "BoundaryLayer",
    "Composite",
    "shear_profile",
]
