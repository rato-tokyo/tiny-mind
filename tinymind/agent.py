"""
TinyMind Agent: Integrated mouse brain-inspired AI system with continuous learning

This module provides the main TinyMindAgent class that combines the three
specialized cortices with continuous unsupervised and reinforcement learning.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .cortex import VisualCortex, LoopCortex, RLCortex
from .config import AGENT, ENVIRONMENT


class TinyMindAgent:
    """
    Integrated TinyMind agent with continuous learning
    
    Architecture:
    - VisualCortex: 25→8→25 (continuous unsupervised learning)
    - LoopCortex: 16→8→16 (continuous unsupervised learning + feedback)
    - RLCortex: DQN (reward-based learning)
    
    Data Flow:
    5x5 Visual Input → VisualCortex → LoopCortex → RLCortex → Actions
    """
    
    def __init__(self):
        """Initialize TinyMind agent with continuous learning cortices"""
        
        # Initialize cortices with continuous learning (using config)
        self.visual_cortex = VisualCortex()  # Uses config defaults
        self.loop_cortex = LoopCortex()      # Uses config defaults  
        self.rl_cortex = RLCortex()          # Uses config defaults
        
        # Learning state
        self.continuous_learning = AGENT["continuous_learning"]
        self.step_count = 0
        
        # Store last temporal features for reward update
        self.last_temporal_features = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.episode_count = 0
    
    def act(self, visual_input: np.ndarray, reward: float = 0.0, done: bool = False) -> int:
        """
        Select action with continuous learning
        
        Args:
            visual_input: Visual field as numpy array (from config)
            reward: Current reward (for SARSA learning)
            done: Whether episode is done
            
        Returns:
            Selected action (0-6 for Minigrid actions)
        """
        # Get visual field dimensions from config
        visual_h, visual_w = ENVIRONMENT["visual_field_size"]
        expected_flat_size = ENVIRONMENT["observation_dim"]
        
        # Ensure correct shape
        if visual_input.shape != (visual_h, visual_w):
            if visual_input.shape == (expected_flat_size,):
                visual_input = visual_input.reshape(visual_h, visual_w)
            else:
                raise ValueError(f"Visual input must be shape ({visual_h},{visual_w}) or ({expected_flat_size},), got {visual_input.shape}")
        
        # Forward pass with continuous learning
        # VisualCortex: continuous unsupervised learning
        visual_features = self.visual_cortex.process(visual_input)
        
        # LoopCortex: continuous unsupervised learning + temporal feedback
        temporal_features = self.loop_cortex.process(visual_features)
        
        # RLCortex: SARSA learning with reward
        action = self.rl_cortex.process(temporal_features, reward=reward, done=done)
        
        # Update tracking
        self.step_count += 1
        self.last_temporal_features = temporal_features.copy()
        self.last_reward = reward
        self.total_reward += reward
        
        return action
    
    def update_reward(self, reward: float, done: bool = False):
        """
        Update RL cortex with reward (for continuous RL learning)
        
        Args:
            reward: Reward received from environment
            done: Whether episode is done
        """
        # Note: For full online RL learning, we would need to implement
        # experience replay and online updates. For now, this is a placeholder
        # that could be extended with custom online learning logic.
        pass
    
    def set_learning(self, enabled: bool):
        """Enable/disable continuous learning for all cortices"""
        self.continuous_learning = enabled
        self.visual_cortex.set_learning(enabled)
        self.loop_cortex.set_learning(enabled)
    
    def reset_memory(self) -> None:
        """Reset memory states of all cortices"""
        self.loop_cortex.reset_memory()
        self.rl_cortex.reset_episode()
        
        # Update episode tracking
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the TinyMind system
        
        Returns:
            Dictionary containing information about all cortices
        """
        rl_stats = self.rl_cortex.get_stats()
        
        return {
            "visual_cortex": self.visual_cortex.get_info(),
            "loop_cortex": self.loop_cortex.get_info(), 
            "rl_cortex": rl_stats,
            "learning_progress": {
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "total_reward": f"{self.total_reward:.3f}",
                "last_reward": f"{self.last_reward:.3f}",
                "epsilon": f"{rl_stats.get('epsilon', 0):.4f}",
                "avg_reward_per_step": f"{self.total_reward / max(1, self.step_count):.4f}"
            },
            "configuration": {
                "continuous_learning": self.continuous_learning,
                "data_flow": AGENT["data_flow"],
                "learning_type": AGENT["learning_paradigm"],
                "architecture_summary": AGENT["architecture_summary"]
            }
        }
    
    def save(self, path: str):
        """Save trained models"""
        try:
            self.rl_cortex.save(path)
            print(f"✅ Model saved to {path}")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def load(self, path: str):
        """Load trained models"""
        try:
            self.rl_cortex.load(path)
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    # Legacy training methods (for backward compatibility)
    def train(self, visual_data: List[np.ndarray] = None, 
              sequences: List[np.ndarray] = None, 
              rl_timesteps: int = 1000):
        """
        Legacy training method - now just trains RL cortex
        Visual and Loop cortices learn continuously during operation
        """
        print("Note: Visual and Loop cortices now learn continuously during operation")
        print("Training RL cortex...")
        self.rl_cortex.train(rl_timesteps)
        print("✅ RL cortex training completed!") 