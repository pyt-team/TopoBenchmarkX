"""Wrappers for cell-based neural networks."""

from .can_wrapper import CANWrapper
from .cccn_wrapper import CCCNWrapper
from .ccxn_wrapper import CCXNWrapper
from .cwn_wrapper import CWNWrapper
from .tune_wrapper import TuneWrapper

__all__ = [
    "CANWrapper",
    "CCCNWrapper",
    "CWNWrapper",
    "CCXNWrapper",
    "TuneWrapper",
    # "OtherWrapper1",
    # "OtherWrapper2",
    # ... add other readout classes here
]
