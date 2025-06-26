"""
TinyMind Environment: Simplified Minigrid-based environments

Using Farama Foundation's Minigrid for biological agent simulation.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
import minigrid
from minigrid.core.constants import COLOR_NAMES
from minigrid.minigrid_env import MiniGridEnv
from ..config import ENVIRONMENT


def make_tinymind_env(env_name: str = "Empty", size: int = None) -> gym.Env:
    """
    Create a TinyMind-compatible environment using Minigrid
    
    Args:
        env_name: Type of environment ("Empty", "FourRooms", "DoorKey")
        size: Grid size (uses config default if None)
        
    Returns:
        Gymnasium environment
    """
    # Use grid size from config if not specified
    if size is None:
        grid_width, grid_height = ENVIRONMENT["grid_size"]
        size = grid_width  # Assume square grid
    
    if env_name == "Empty":
        env_id = f"MiniGrid-Empty-{size}x{size}-v0"
    elif env_name == "FourRooms":
        env_id = "MiniGrid-FourRooms-v0"
    elif env_name == "DoorKey":
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
    else:
        env_id = f"MiniGrid-Empty-{size}x{size}-v0"
    
    try:
        env = gym.make(env_id)
        return TinyMindWrapper(env)
    except:
        # Fallback to simple empty environment
        env = gym.make("MiniGrid-Empty-5x5-v0")
        return TinyMindWrapper(env)


class TinyMindWrapper(gym.Wrapper):
    """
    Wrapper to adapt Minigrid environments for TinyMind agents
    
    Provides simplified observation space and reward structure
    suitable for small biological agents
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Simplified observation space - flattened visual field (from config)
        visual_size = ENVIRONMENT["visual_field_size"]
        obs_dim = ENVIRONMENT["observation_dim"]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        self.visual_field_size = visual_size
        
        # Keep original action space (turn left, turn right, move forward, etc.)
        self.action_space = env.action_space
        
        self.step_count = 0
        self.max_steps = ENVIRONMENT["max_steps"]
        
    def reset(self, **kwargs):
        """Reset environment and return simplified observation"""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return self._process_observation(obs), info
    
    def step(self, action):
        """Take step and return simplified observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Improved reward structure for better SARSA learning (from config)
        rewards = ENVIRONMENT["rewards"]
        
        # Very small step penalty to encourage efficiency
        reward += rewards["step_penalty"]
        
        # Significant exploration bonus for moving (encourage exploration)
        if action in [2]:  # Forward movement
            reward += rewards["forward_bonus"]
        
        # Turn bonuses (exploration in different directions)
        if action in [0, 1]:  # Turn left, turn right
            reward += rewards["turn_bonus"]
        
        # Survival bonus (staying alive is valuable)
        if not terminated:
            reward += rewards["survival_bonus"]
            
        # Progress milestones with increasing rewards
        if self.step_count % 100 == 0 and not terminated:
            reward += rewards["milestone_100"]
        elif self.step_count % 50 == 0 and not terminated:
            reward += rewards["milestone_50"]
            
        # Bonus for longer survival (exponential survival value)
        if self.step_count > rewards["long_survival_start"] and not terminated:
            reward += rewards["long_survival_factor"] * (self.step_count / 1000)
        
        # Terminate if too many steps (biological energy constraint)
        if self.step_count >= self.max_steps:
            truncated = True
            # Less harsh penalty for timeout
            reward += rewards["timeout_penalty"]
        
        return self._process_observation(obs), reward, terminated, truncated, info
    
    def _process_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Convert Minigrid observation to simplified visual field (from config)
        
        Args:
            obs: Minigrid observation dictionary
            
        Returns:
            Flattened visual field as numpy array
        """
        # Extract image from observation
        if isinstance(obs, dict) and 'image' in obs:
            image = obs['image']
        else:
            # Fallback for direct image observations
            image = obs
        
        # Get visual field size from config
        visual_h, visual_w = self.visual_field_size
        
        # Resize to visual field size if needed
        if image.shape[:2] != (visual_h, visual_w):
            # Simple downsampling to configured size
            h, w = image.shape[:2]
            step_h, step_w = h // visual_h, w // visual_w
            resized = np.zeros((visual_h, visual_w, image.shape[2] if len(image.shape) > 2 else 1))
            
            for i in range(visual_h):
                for j in range(visual_w):
                    start_h, end_h = i * step_h, min((i + 1) * step_h, h)
                    start_w, end_w = j * step_w, min((j + 1) * step_w, w)
                    
                    if len(image.shape) > 2:
                        resized[i, j] = np.mean(image[start_h:end_h, start_w:end_w], axis=(0, 1))
                    else:
                        resized[i, j, 0] = np.mean(image[start_h:end_h, start_w:end_w])
            
            image = resized
        
        # Convert to grayscale and normalize
        if len(image.shape) > 2:
            # Convert RGB to grayscale
            gray = np.mean(image, axis=2)
        else:
            gray = image.squeeze()
        
        # Normalize to [0, 1]
        if gray.max() > 1:
            gray = gray / gray.max()
        
        # Flatten to configured dimension vector
        return gray.flatten().astype(np.float32)


# Available environment configurations
TINYMIND_ENVIRONMENTS = {
    "empty": {
        "name": "Empty",
        "size": 5,
        "description": "Simple empty grid for basic navigation"
    },
    "small": {
        "name": "Empty", 
        "size": 7,
        "description": "Slightly larger empty grid"
    },
    "rooms": {
        "name": "FourRooms",
        "size": 9,
        "description": "Four rooms environment for exploration"
    },
    "key": {
        "name": "DoorKey",
        "size": 5,
        "description": "Door and key environment for goal-directed behavior (more rewards)"
    },
    "hunting": {
        "name": "Empty",
        "size": 8,
        "description": "Larger hunting ground with more exploration opportunities"
    }
} 