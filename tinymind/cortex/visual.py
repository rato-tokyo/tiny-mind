"""
Visual Cortex - Single Autoencoder with continuous unsupervised learning

Implements continuous autoencoder learning: 25→8→25
"""
import torch
import torch.nn as nn
import numpy as np
from .base import AutoencoderLayer
from ..config import VISUAL_CORTEX, ENVIRONMENT


class VisualCortex:
    """
    Visual cortex using single autoencoder with continuous learning
    
    Architecture: 25 → 8 → 25
    Learning: Continuous unsupervised learning during processing
    """
    
    def __init__(self, input_dim: int = None, hidden_dim: int = None, learning_rate: float = None):
        # Load from config if not specified
        self.input_dim = input_dim or VISUAL_CORTEX["input_dim"]
        self.hidden_dim = hidden_dim or VISUAL_CORTEX["hidden_dim"]
        self.learning_rate = learning_rate or VISUAL_CORTEX["learning_rate"]
        
        # Single autoencoder layer: 25 → 8 → 25
        self.autoencoder = AutoencoderLayer(self.input_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Learning state
        self.learning_enabled = True
        self.learning_count = 0
        self.current_loss = 0.0
    
    def process(self, visual_input: np.ndarray) -> np.ndarray:
        """Process visual input with continuous learning (size from config)"""
        # Get visual field size from config
        visual_h, visual_w = ENVIRONMENT["visual_field_size"]
        expected_size = ENVIRONMENT["observation_dim"]
        
        # Flatten visual input to expected dimensional vector
        if visual_input.shape == (visual_h, visual_w):
            visual_input = visual_input.flatten()
        elif visual_input.size == expected_size:
            visual_input = visual_input.flatten()
        else:
            raise ValueError(f"Expected visual input size {expected_size} or shape ({visual_h},{visual_w}), got shape {visual_input.shape}")
        
        # Convert to tensor
        x = torch.FloatTensor(visual_input).unsqueeze(0)
        
        # Continuous unsupervised learning
        if self.learning_enabled:
            # Forward pass for reconstruction
            reconstructed = self.autoencoder(x)
            loss = self.criterion(reconstructed, x)
            
            # Store current loss for monitoring
            self.current_loss = loss.item()
            
            # Backward pass (online learning)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.learning_count += 1
        
        # Extract features (encoding)
        with torch.no_grad():
            features = self.autoencoder.encode(x)
            return features.numpy().flatten()
    
    def set_learning(self, enabled: bool):
        """Enable/disable continuous learning"""
        self.learning_enabled = enabled
    
    def get_info(self) -> dict:
        """Get cortex information"""
        return {
            "type": "VisualCortex",
            "algorithm": "Single Autoencoder (Continuous Learning)",
            "architecture": VISUAL_CORTEX["architecture"],
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.hidden_dim,
            "learning_enabled": self.learning_enabled,
            "learning_count": self.learning_count,
            "training_loss": self.current_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        } 