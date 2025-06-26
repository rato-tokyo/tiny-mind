"""
Cortex module containing the three specialized brain regions
"""

from .visual import VisualCortex
from .loop import LoopCortex
from .rl import RLCortex

__all__ = ["VisualCortex", "LoopCortex", "RLCortex"] 