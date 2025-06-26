#!/usr/bin/env python3
"""
TinyMind Basic Usage Example

This script demonstrates the basic usage of the TinyMind agent,
including initialization, action selection, and learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tinymind import TinyMindAgent


def main():
    print("🧠 TinyMind Basic Usage Example")
    print("=" * 50)
    
    # Initialize TinyMind agent
    print("Initializing TinyMind agent...")
    agent = TinyMindAgent(
        visual_dims=[25, 16, 8],    # Visual cortex dimensions
        memory_dim=8,               # Memory dimension
        learning_rates={
            "visual": 0.001,
            "loop": 0.001,
            "rl": 0.001
        }
    )
    
    # Display agent information
    info = agent.get_info()
    print(f"✅ Agent initialized successfully!")
    print(f"   Total parameters: {info['total_parameters']:,}")
    print(f"   Visual cortex: {info['visual_parameters']:,} params")
    print(f"   Loop cortex: {info['loop_parameters']:,} params")
    print(f"   RL cortex: {info['rl_parameters']:,} params")
    print()
    
    # Example 1: Basic action selection
    print("📊 Example 1: Basic Action Selection")
    print("-" * 30)
    
    # Create a random 5x5 visual input
    visual_input = torch.randn(5, 5)
    print(f"Visual input shape: {visual_input.shape}")
    
    # Select action (inference mode)
    action = agent.act(visual_input, training=False)
    print(f"Selected action: {action} (out of 0-8)")
    
    # Get Q-values for all actions
    q_values = agent.get_q_values(visual_input.flatten())
    print(f"Q-values: {q_values.detach().numpy()}")
    
    # Get action probabilities
    action_probs = agent.get_action_probabilities(visual_input.flatten())
    print(f"Action probabilities: {action_probs.detach().numpy()}")
    print()
    
    # Example 2: Learning from experience
    print("🎓 Example 2: Learning from Experience")
    print("-" * 30)
    
    # Simulate a simple learning episode
    total_reward = 0
    episode_losses = {"visual": [], "loop": [], "rl": []}
    
    for step in range(50):
        # Current state
        current_visual = torch.randn(5, 5)
        
        # Select action (training mode with exploration)
        action = agent.act(current_visual, training=True)
        
        # Simulate environment response
        next_visual = torch.randn(5, 5)
        reward = np.random.randn()  # Random reward for demo
        done = step >= 49  # End episode after 50 steps
        
        # Learn from experience
        losses = agent.learn(
            visual_input=current_visual,
            action=action,
            reward=reward,
            next_visual_input=next_visual,
            done=done
        )
        
        # Store losses for analysis
        for key, value in losses.items():
            if value is not None:
                episode_losses[key].append(value)
        
        total_reward += reward
        
        if step % 10 == 0:
            current_epsilon = agent.rl_cortex.get_current_epsilon()
            print(f"Step {step:2d}: Action={action}, Reward={reward:+.3f}, ε={current_epsilon:.3f}")
    
    print(f"Episode completed! Total reward: {total_reward:.3f}")
    print()
    
    # Example 3: Memory state analysis
    print("🧠 Example 3: Memory State Analysis")
    print("-" * 30)
    
    # Reset memory and observe changes
    print("Before reset:")
    memory_before = agent.get_memory_state()
    print(f"Memory state: {memory_before.detach().numpy()}")
    
    agent.reset_memory()
    
    print("After reset:")
    memory_after = agent.get_memory_state()
    print(f"Memory state: {memory_after.detach().numpy()}")
    
    # Process a sequence of inputs to see memory evolution
    print("\nMemory evolution during sequence processing:")
    for i in range(3):
        visual_input = torch.randn(5, 5)
        action = agent.act(visual_input, training=False)
        memory_state = agent.get_memory_state()
        print(f"Step {i+1}: Memory = {memory_state.detach().numpy()}")
    
    print()
    
    # Example 4: Model saving and loading
    print("💾 Example 4: Model Saving and Loading")
    print("-" * 30)
    
    # Save the model
    save_path = "example_tinymind_model"
    print(f"Saving model to '{save_path}'...")
    agent.save(save_path)
    print("✅ Model saved successfully!")
    
    # Create a new agent and load the saved model
    print("Creating new agent and loading saved model...")
    new_agent = TinyMindAgent()
    new_agent.load(save_path)
    print("✅ Model loaded successfully!")
    
    # Verify that the loaded model produces the same output
    test_input = torch.randn(5, 5)
    original_action = agent.act(test_input, training=False)
    loaded_action = new_agent.act(test_input, training=False)
    
    print(f"Original agent action: {original_action}")
    print(f"Loaded agent action: {loaded_action}")
    print(f"Actions match: {original_action == loaded_action}")
    print()
    
    # Summary
    print("🎉 Example completed successfully!")
    print("Key takeaways:")
    print("- TinyMind integrates three specialized cortices")
    print("- Easy action selection with act() method")
    print("- Learning from experience with learn() method")
    print("- Memory state management for temporal tasks")
    print("- Simple model saving and loading")
    print("\nFor more advanced examples, check the examples/ directory!")


if __name__ == "__main__":
    main() 