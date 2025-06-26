"""
TinyMind Environments

Standard environments for training and testing TinyMind agents.
"""

from .hunting_env import make_tinymind_env, TinyMindWrapper, TINYMIND_ENVIRONMENTS

__all__ = ["make_tinymind_env", "TinyMindWrapper", "TINYMIND_ENVIRONMENTS"]
