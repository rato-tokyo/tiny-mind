"""
TinyMind Configuration - Central parameter management

All system parameters are defined here for easy management and tuning.
"""

# ===== VISUAL CORTEX CONFIGURATION =====
VISUAL_CORTEX = {
    "visual_field_size": (6, 6),  # Visual field dimensions
    "input_dim": 36,           # 6x6 visual field flattened
    "hidden_dim": 30,          # Compressed feature dimension (increased)
    "output_dim": 30,          # Same as hidden_dim for autoencoder
    "learning_rate": 0.001,    # Learning rate for Adam optimizer
    "architecture": "36→30→36", # Visual representation
}

# ===== LOOP CORTEX CONFIGURATION =====
LOOP_CORTEX = {
    "visual_input_dim": 30,    # Input from VisualCortex (updated)
    "feedback_dim": 20,        # Previous output feedback (increased capacity)
    "total_input_dim": 50,     # visual_input_dim + feedback_dim
    "hidden_dim": 20,          # Compressed temporal feature dimension (increased)
    "output_dim": 20,          # Temporal features output (increased)
    "learning_rate": 0.0001,   # Lower learning rate to prevent gradient explosion
    "memory_size": 100,        # Maximum memory buffer size
    "feedback_decay": 0.9,     # Decay factor for temporal feedback (0.0-1.0)
    "gradient_clip_norm": 1.0, # Maximum gradient norm for clipping
    "output_clip_range": (-1.0, 1.0),  # Output value clipping range
    "architecture": "50→20→50", # Architecture representation
}

# ===== RL CORTEX CONFIGURATION (SARSA) =====
RL_CORTEX = {
    "input_dim": 20,           # Input from LoopCortex (updated to match new output)
    "action_dim": 7,           # Minigrid standard actions
    "hidden_dim": 32,          # Hidden layer size for Q-network
    "learning_rate": 0.01,     # SARSA learning rate
    "gamma": 0.99,             # Discount factor
    "epsilon_start": 0.3,      # Initial exploration rate
    "epsilon_decay": 0.9995,   # Epsilon decay rate per step
    "epsilon_min": 0.05,       # Minimum exploration rate
    "device": "cpu",           # Computing device
    "algorithm": "SARSA",      # Algorithm name
}

# ===== ENVIRONMENT CONFIGURATION =====
ENVIRONMENT = {
    "max_steps": 5000,         # Maximum steps per episode (10x increased)
    "grid_size": (10, 10),     # Overall environment grid size
    "visual_field_size": (6, 6),  # Agent's visual observation window (subset of grid)
    "observation_dim": 36,     # Flattened visual field (updated)
    
    # Reward structure (optimized for learning)
    "rewards": {
        "step_penalty": -0.0001,      # Very small step penalty
        "forward_bonus": 0.02,        # Reward for moving forward
        "turn_bonus": 0.005,          # Reward for turning (exploration)
        "survival_bonus": 0.005,      # Reward for staying alive
        "milestone_50": 0.02,         # Bonus every 50 steps
        "milestone_100": 0.05,        # Bonus every 100 steps
        "timeout_penalty": -0.1,      # Penalty for timeout
        "long_survival_start": 200,   # Steps when long survival bonus starts
        "long_survival_factor": 0.01, # Long survival bonus factor
    }
}

# ===== AGENT CONFIGURATION =====
AGENT = {
    "continuous_learning": True,   # Enable continuous learning for all cortices
    "data_flow": "6×6 Visual → VisualCortex → LoopCortex → RLCortex → 7 Actions",
    "learning_paradigm": "Continuous Online Learning",
    "architecture_summary": {
        "visual_cortex": VISUAL_CORTEX["architecture"],
        "loop_cortex": LOOP_CORTEX["architecture"], 
        "rl_cortex": f"{RL_CORTEX['input_dim']}→{RL_CORTEX['action_dim']} (SARSA)",
    }
}

# ===== CLI CONFIGURATION =====
CLI = {
    "default_episodes": 10,        # Default number of episodes for demo
    "default_env": "empty",        # Default environment type
    "verbose_analysis_interval": 5, # Show detailed analysis every N episodes
    "learning_improvement_threshold": 0.01,  # Threshold for learning success
}

# ===== SYSTEM INFORMATION =====
SYSTEM_INFO = {
    "name": "TinyMind",
    "version": "1.0.0",
    "description": "Biologically-inspired minimal AI system with continuous learning",
    "python_requirement": ">=3.10",
    "key_features": [
        "Continuous online learning",
        "SARSA-based reinforcement learning", 
        "Lightweight autoencoder cortices",
        "Biological mouse brain architecture",
        "Real-time learning analysis"
    ],
    "total_parameters": {
        "visual_cortex": VISUAL_CORTEX["input_dim"] * VISUAL_CORTEX["hidden_dim"] * 2,  # Encoder + Decoder
        "loop_cortex": LOOP_CORTEX["total_input_dim"] * LOOP_CORTEX["hidden_dim"] * 2,  # Encoder + Decoder  
        "rl_cortex": (RL_CORTEX["input_dim"] * RL_CORTEX["hidden_dim"] + 
                     RL_CORTEX["hidden_dim"] * RL_CORTEX["hidden_dim"] + 
                     RL_CORTEX["hidden_dim"] * RL_CORTEX["action_dim"])  # Q-network layers
    }
}

# Calculate total system parameters
SYSTEM_INFO["total_parameters"]["total"] = sum(SYSTEM_INFO["total_parameters"].values())

# ===== VALIDATION FUNCTIONS =====
def validate_config():
    """Validate configuration consistency"""
    errors = []
    
    # Check dimension compatibility
    if VISUAL_CORTEX["output_dim"] != LOOP_CORTEX["visual_input_dim"]:
        errors.append(f"VisualCortex output ({VISUAL_CORTEX['output_dim']}) != LoopCortex visual input ({LOOP_CORTEX['visual_input_dim']})")
    
    if LOOP_CORTEX["output_dim"] != RL_CORTEX["input_dim"]:
        errors.append(f"LoopCortex output ({LOOP_CORTEX['output_dim']}) != RLCortex input ({RL_CORTEX['input_dim']})")
    
    if LOOP_CORTEX["visual_input_dim"] + LOOP_CORTEX["feedback_dim"] != LOOP_CORTEX["total_input_dim"]:
        errors.append(f"LoopCortex input dimensions don't add up: {LOOP_CORTEX['visual_input_dim']} + {LOOP_CORTEX['feedback_dim']} != {LOOP_CORTEX['total_input_dim']}")
    
    # Check learning rates
    if LOOP_CORTEX["learning_rate"] >= VISUAL_CORTEX["learning_rate"]:
        errors.append("LoopCortex learning rate should be lower than VisualCortex to prevent gradient explosion")
    
    # Check epsilon values
    if RL_CORTEX["epsilon_min"] >= RL_CORTEX["epsilon_start"]:
        errors.append(f"Epsilon min ({RL_CORTEX['epsilon_min']}) should be < epsilon start ({RL_CORTEX['epsilon_start']})")
    
    return errors

def get_config_summary():
    """Get a human-readable configuration summary"""
    summary = {
        "System": SYSTEM_INFO["name"] + " v" + SYSTEM_INFO["version"],
        "Architecture": AGENT["data_flow"],
        "Learning": AGENT["learning_paradigm"],
        "Total Parameters": f"{SYSTEM_INFO['total_parameters']['total']:,}",
        "Visual Cortex": f"{VISUAL_CORTEX['architecture']} (lr={VISUAL_CORTEX['learning_rate']})",
        "Loop Cortex": f"{LOOP_CORTEX['architecture']} (lr={LOOP_CORTEX['learning_rate']}, decay={LOOP_CORTEX['feedback_decay']})",
        "RL Cortex": f"{RL_CORTEX['algorithm']} (lr={RL_CORTEX['learning_rate']}, ε={RL_CORTEX['epsilon_start']}→{RL_CORTEX['epsilon_min']})",
        "Environment": f"Grid: {ENVIRONMENT['grid_size']}, Visual: {ENVIRONMENT['visual_field_size']}, Max steps: {ENVIRONMENT['max_steps']}"
    }
    return summary

# Run validation on import
_validation_errors = validate_config()
if _validation_errors:
    print("⚠️  Configuration validation warnings:")
    for error in _validation_errors:
        print(f"   - {error}") 