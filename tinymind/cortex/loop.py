"""
Loop Cortex - Single Autoencoder with continuous learning and feedback

Implements continuous autoencoder learning with temporal feedback: 16→8→16
Input: previous_output(8) + visual_features(8) = 16 dimensions
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List
from collections import deque
from .base import AutoencoderLayer
from ..config import LOOP_CORTEX


class LoopCortex:
    """
    Loop cortex using single autoencoder with continuous learning and feedback
    
    Input: previous_output(8) + visual_features(8) = 16 dimensions
    Architecture: 16 → 8 → 16
    Learning: Continuous unsupervised learning during processing
    """
    
    def __init__(self, input_dim: int = None, hidden_dim: int = None, memory_size: int = None, learning_rate: float = None):
        # Load from config if not specified
        self.input_dim = input_dim or LOOP_CORTEX["visual_input_dim"]
        self.hidden_dim = hidden_dim or LOOP_CORTEX["hidden_dim"]
        self.memory_size = memory_size or LOOP_CORTEX["memory_size"]
        self.learning_rate = learning_rate or LOOP_CORTEX["learning_rate"]
        self.feedback_decay = LOOP_CORTEX["feedback_decay"]
        self.gradient_clip_norm = LOOP_CORTEX["gradient_clip_norm"]
        self.output_clip_range = LOOP_CORTEX["output_clip_range"]
        
        # Single autoencoder: 16 → 8 → 16
        self.autoencoder = AutoencoderLayer(LOOP_CORTEX["total_input_dim"], self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Memory and state
        self.memory = deque(maxlen=self.memory_size)
        self.previous_output = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Learning state
        self.learning_enabled = True
        self.learning_count = 0
        self.current_loss = 0.0
        
    def process(self, visual_features: np.ndarray) -> np.ndarray:
        """Process visual features with temporal memory feedback and continuous learning"""
        # Ensure visual_features matches expected dimension
        expected_dim = LOOP_CORTEX["visual_input_dim"]
        if len(visual_features) != expected_dim:
            if len(visual_features) < expected_dim:
                visual_features = np.pad(visual_features, (0, expected_dim-len(visual_features)))
            else:
                visual_features = visual_features[:expected_dim]
        
        # Combine with previous output: [previous_output(8) + visual_features(8)] = 16
        combined_input = np.concatenate([self.previous_output, visual_features])
        
        # Normalize input to prevent gradient explosion
        combined_input = combined_input / (np.linalg.norm(combined_input) + 1e-8)
        
        x = torch.FloatTensor(combined_input).unsqueeze(0)
        
        # Continuous unsupervised learning
        if self.learning_enabled:
            # Forward pass for reconstruction
            reconstructed = self.autoencoder(x)
            loss = self.criterion(reconstructed, x)
            
            # Store current loss for monitoring
            self.current_loss = loss.item()
            
            # Backward pass (online learning) with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=self.gradient_clip_norm)
            
            self.optimizer.step()
            
            self.learning_count += 1
        
        # Extract features (encoding) for output
        with torch.no_grad():
            output = self.autoencoder.encode(x).numpy().flatten()
            
            # Clip output to prevent explosion
            output = np.clip(output, *self.output_clip_range)
        
        # Store for next iteration (temporal feedback) with decay
        self.previous_output = output.copy() * self.feedback_decay
        self.memory.append(output.copy())
        
        return output
    
    def set_learning(self, enabled: bool):
        """Enable/disable continuous learning"""
        self.learning_enabled = enabled
    
    def reset_memory(self):
        """Reset temporal memory"""
        self.memory.clear()
        self.previous_output = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def get_info(self) -> dict:
        """Get cortex information"""
        return {
            "type": "LoopCortex",
            "algorithm": "Single Autoencoder with Feedback (Continuous Learning)",
            "architecture": LOOP_CORTEX["architecture"],
            "input_dim": LOOP_CORTEX["total_input_dim"],
            "hidden_dim": self.hidden_dim,
            "output_dim": self.hidden_dim,
            "memory_size": len(self.memory),
            "learning_enabled": self.learning_enabled,
            "learning_count": self.learning_count,
            "training_loss": self.current_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        } 