"""
RL Cortex - SARSA-based continuous learning implementation

Lightweight SARSA implementation optimized for continuous online learning.
Compatible with the TinyMind continuous learning architecture.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple
import random
from ..config import RL_CORTEX


class SARSANetwork(nn.Module):
    """Simple Q-network for SARSA learning"""
    
    def __init__(self, input_dim: int = 8, action_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RLCortex:
    """
    RL Cortex using SARSA algorithm for continuous learning
    
    SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference learning
    algorithm that updates Q-values based on the actual action taken in the next state.
    """
    
    def __init__(
        self,
        input_dim: int = None,
        action_dim: int = None,
        learning_rate: float = None,
        gamma: float = None,
        epsilon: float = None,
        epsilon_decay: float = None,
        epsilon_min: float = None,
        device: str = None
    ):
        # Load from config if not specified
        self.input_dim = input_dim or RL_CORTEX["input_dim"]
        self.action_dim = action_dim or RL_CORTEX["action_dim"]
        self.gamma = gamma or RL_CORTEX["gamma"]
        self.epsilon = epsilon or RL_CORTEX["epsilon_start"]
        self.epsilon_decay = epsilon_decay or RL_CORTEX["epsilon_decay"]
        self.epsilon_min = epsilon_min or RL_CORTEX["epsilon_min"]
        self.device = torch.device(device or RL_CORTEX["device"])
        self.learning_rate = learning_rate or RL_CORTEX["learning_rate"]
        
        # Q-network for SARSA
        self.q_network = SARSANetwork(self.input_dim, self.action_dim, RL_CORTEX["hidden_dim"]).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Store previous state-action for SARSA update
        self.prev_state = None
        self.prev_action = None
        self.prev_q_value = None
        
        print(f"RLCortex initialized with SARSA algorithm")
        print(f"Input dim: {self.input_dim}, Action dim: {self.action_dim}")
        print(f"Learning rate: {self.learning_rate}, Gamma: {self.gamma}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Input state from LoopCortex (8 dimensions)
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index (0-6 for Minigrid)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
        
        return action
    
    def process(self, loop_output: np.ndarray, reward: float = 0.0, done: bool = False) -> int:
        """
        Process input from LoopCortex and return action with continuous learning
        
        Args:
            loop_output: 8-dimensional output from LoopCortex
            reward: Reward from environment (for SARSA update)
            done: Whether episode is finished
            
        Returns:
            Action index (0-6)
        """
        # Ensure correct input dimensions
        if len(loop_output) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {len(loop_output)}")
        
        current_state = torch.FloatTensor(loop_output).unsqueeze(0).to(self.device)
        
        # Select current action
        current_action = self.select_action(loop_output, training=True)
        
        # SARSA learning: update Q(s,a) using Q(s',a')
        if self.prev_state is not None and self.prev_action is not None:
            self._sarsa_update(current_state, current_action, reward, done)
        
        # Store current state-action for next update
        self.prev_state = current_state.clone()
        self.prev_action = current_action
        
        # Get Q-value for the selected action
        with torch.no_grad():
            q_values = self.q_network(current_state)
            self.prev_q_value = q_values[0, current_action].item()
        
        # Decay epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return current_action
    
    def _sarsa_update(self, current_state: torch.Tensor, current_action: int, 
                     reward: float, done: bool):
        """
        Perform SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Args:
            current_state: Current state s'
            current_action: Current action a'
            reward: Reward r received
            done: Whether episode is finished
        """
        # Get current Q-value Q(s',a')
        current_q_values = self.q_network(current_state)
        current_q_value = current_q_values[0, current_action]
        
        # Get previous Q-value Q(s,a)
        prev_q_values = self.q_network(self.prev_state)
        prev_q_value = prev_q_values[0, self.prev_action]
        
        # SARSA target: r + γQ(s',a') (or just r if done)
        if done:
            target = reward
        else:
            target = reward + self.gamma * current_q_value.item()
        
        # SARSA loss: [target - Q(s,a)]²
        target_tensor = torch.FloatTensor([target]).to(self.device)
        loss = F.mse_loss(prev_q_value.unsqueeze(0), target_tensor)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def reset_episode(self):
        """Reset episode-specific state"""
        self.prev_state = None
        self.prev_action = None
        self.prev_q_value = None
    
    def get_state_dict(self) -> dict:
        """Get model state for saving"""
        return {
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load model state"""
        self.q_network.load_state_dict(state_dict['q_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epsilon = state_dict.get('epsilon', self.epsilon)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for debugging"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.cpu().numpy().flatten()
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        # Get current Q-values for analysis
        if self.prev_state is not None:
            current_q_values = self.get_q_values(self.prev_state.cpu().numpy().flatten())
            max_q = np.max(current_q_values)
            min_q = np.min(current_q_values)
            avg_q = np.mean(current_q_values)
        else:
            max_q = min_q = avg_q = 0.0
            
        return {
            'algorithm': 'SARSA',
            'epsilon': self.epsilon,
            'prev_q_value': self.prev_q_value,
            'network_params': sum(p.numel() for p in self.q_network.parameters()),
            'q_value_stats': {
                'max_q': float(max_q),
                'min_q': float(min_q),
                'avg_q': float(avg_q),
            }
        } 