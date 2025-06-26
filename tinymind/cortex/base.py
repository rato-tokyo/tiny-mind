"""
Base classes for cortex implementations
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseCortex(nn.Module, ABC):
    """
    Base class for all cortical regions
    """
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the cortex"""
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the appropriate device"""
        return tensor.to(self.device)


class AutoencoderLayer(nn.Module):
    """Single autoencoder layer"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class StagedAutoencoder:
    """Base class for staged multi-layer autoencoder"""
    
    def __init__(self, layer1_config: Tuple[int, int], layer2_config: Tuple[int, int]):
        """
        Args:
            layer1_config: (input_dim, hidden_dim) for layer 1
            layer2_config: (input_dim, hidden_dim) for layer 2
        """
        # Layer 1
        self.layer1 = AutoencoderLayer(*layer1_config)
        self.optimizer1 = torch.optim.Adam(self.layer1.parameters(), lr=0.001)
        self.criterion1 = nn.MSELoss()
        
        # Layer 2
        self.layer2 = AutoencoderLayer(*layer2_config)
        self.optimizer2 = torch.optim.Adam(self.layer2.parameters(), lr=0.001)
        self.criterion2 = nn.MSELoss()
        
        self.layer1_trained = False
        self.layer2_trained = False
    
    def _train_layer(self, layer, optimizer, criterion, data, epochs, layer_name):
        """Generic layer training method"""
        print(f"Training {layer_name}...")
        
        for epoch in range(epochs):
            total_loss = 0
            for sample in data:
                x = torch.FloatTensor(sample).unsqueeze(0)
                
                reconstructed = layer(x)
                loss = criterion(reconstructed, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"{layer_name} Epoch {epoch}, Loss: {total_loss/len(data):.4f}")
        
        print(f"{layer_name} training completed!")
    
    def encode_through_layers(self, x):
        """Encode through both layers if trained"""
        with torch.no_grad():
            if self.layer1_trained:
                features = self.layer1.encode(x)
                if self.layer2_trained:
                    features = self.layer2.encode(features)
                return features
            return None


class AutoEncoder(nn.Module):
    """
    Multi-layer autoencoder for feature extraction and reconstruction
    """
    
    def __init__(self, dims: List[int]):
        """
        Initialize autoencoder with specified dimensions
        
        Args:
            dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super().__init__()
        
        if len(dims) < 2:
            raise ValueError("Need at least input and output dimensions")
        
        self.dims = dims
        
        # Encoder layers
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU() if i < len(dims) - 2 else nn.Identity()  # No activation on last layer
            ])
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        for i in range(len(dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(dims[i], dims[i - 1]),
                nn.ReLU() if i > 1 else nn.Sigmoid()  # Sigmoid for output layer
            ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder
        
        Returns:
            Tuple of (encoded_features, reconstructed_input)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded 