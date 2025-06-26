"""
TinyMind: A biologically-inspired minimal AI system

This package implements a mouse brain-inspired AI architecture with three specialized cortices:
- VisualCortex: Hierarchical feature extraction using multi-layer autoencoders
- LoopCortex: Temporal memory integration with recurrent autoencoders  
- RLCortex: Action selection and learning using Deep Q-Network (DQN)
"""

__version__ = "0.1.0"
__author__ = "TinyMind Development Team"

from .cortex.visual import VisualCortex
from .cortex.loop import LoopCortex
from .cortex.rl import RLCortex
from .agent import TinyMindAgent

__all__ = [
    "VisualCortex",
    "LoopCortex", 
    "RLCortex",
    "TinyMindAgent",
] 