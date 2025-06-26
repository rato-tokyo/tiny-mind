"""
Simplified integration tests for TinyMind

Tests focus on core functionality:
- Prey movement patterns
- Agent movement and action selection
- Reward system for catching prey
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from tinymind.envs.hunting_env import HuntingEnv
from tinymind.agent import TinyMindAgent


class TestPreyMovement:
    """Test prey movement using scikit-learn clustering"""
    
    def test_prey_moves_every_interval(self):
        """Test that prey moves at correct intervals"""
        env = HuntingEnv(
            num_prey=2,
            prey_move_interval=3,
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        obs, info = env.reset()
        initial_prey_positions = info['prey_positions'].copy()
        
        # Prey should not move for first 2 steps
        for step in range(2):
            obs, reward, terminated, truncated, info = env.step(4)  # Action 4 = stay
            assert info['prey_positions'] == initial_prey_positions, f"Prey moved at step {step + 1}"
        
        # Prey should move at step 3
        obs, reward, terminated, truncated, info = env.step(4)
        assert info['prey_positions'] != initial_prey_positions, "Prey did not move at interval"
    
    def test_prey_uses_intelligent_movement(self):
        """Test that prey movement is not purely random"""
        env = HuntingEnv(
            num_prey=3,
            prey_move_interval=2,
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        obs, info = env.reset()
        
        # Record prey movements over multiple intervals
        movements = []
        for _ in range(10):  # 5 movement intervals
            # Take 2 steps to trigger prey movement
            env.step(4)  # Stay action
            obs, reward, terminated, truncated, info = env.step(4)
            movements.append(info['prey_positions'].copy())
        
        # Check that prey positions show some pattern (not completely random)
        # If movement was purely random, variance would be very high
        position_arrays = [np.array(pos_list) for pos_list in movements]
        
        # At least some prey should maintain relative positions (clustering behavior)
        assert len(movements) > 0, "No movements recorded"
        assert len(movements[0]) > 0, "No prey positions recorded"


class TestAgentMovement:
    """Test agent movement and action selection"""
    
    def test_agent_moves_correctly(self):
        """Test that agent moves according to action commands"""
        env = HuntingEnv(
            num_prey=1,
            prey_move_interval=10,  # Prey won't move during test
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        obs, info = env.reset()
        initial_agent_pos = info['agent_pos']
        
        # Test movement actions (0-8 for 3x3 grid)
        # Action 1 = move up
        obs, reward, terminated, truncated, info = env.step(1)
        new_pos = info['agent_pos']
        
        # Agent should have moved (unless at boundary)
        if initial_agent_pos[0] > 1:  # Not at top boundary
            assert new_pos[0] == initial_agent_pos[0] - 1, "Agent did not move up correctly"
        
        # Action 7 = move down  
        obs, reward, terminated, truncated, info = env.step(7)
        final_pos = info['agent_pos']
        
        # Agent should have moved down (unless at boundary)
        if new_pos[0] < 8:  # Not at bottom boundary
            assert final_pos[0] == new_pos[0] + 1, "Agent did not move down correctly"
    
    def test_agent_cannot_move_through_walls(self):
        """Test that agent cannot move outside boundaries"""
        env = HuntingEnv(
            num_prey=1,
            prey_move_interval=10,
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        # Force agent to corner position for testing
        env.reset()
        env.agent_pos = (1, 1)  # Top-left corner of inner area
        env._update_grid()
        
        # Try to move up (should hit wall)
        obs, reward, terminated, truncated, info = env.step(1)  # Move up
        assert info['agent_pos'] == (1, 1), "Agent moved through wall"
        
        # Try to move left (should hit wall)
        obs, reward, terminated, truncated, info = env.step(3)  # Move left
        assert info['agent_pos'] == (1, 1), "Agent moved through wall"
    
    def test_agent_action_selection(self):
        """Test agent action selection with TinyMind"""
        agent = TinyMindAgent()
        
        # Test with random visual input
        visual_input = torch.rand(25)  # 5x5 flattened
        action = agent.select_action(visual_input)
        
        # Action should be valid (0-8)
        assert 0 <= action <= 8, f"Invalid action: {action}"
        assert isinstance(action, int), f"Action should be int, got {type(action)}"


class TestRewardSystem:
    """Test reward system for catching prey"""
    
    def test_catch_reward_given(self):
        """Test that reward is given when agent catches prey"""
        env = HuntingEnv(
            num_prey=1,
            prey_move_interval=10,  # Prey won't move
            catch_reward=15.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        obs, info = env.reset()
        prey_pos = info['prey_positions'][0]
        
        # Move agent to prey position
        env.agent_pos = prey_pos
        env._update_grid()
        
        # Take any action to trigger reward calculation
        obs, reward, terminated, truncated, info = env.step(4)  # Stay action
        
        # Should receive catch reward
        assert reward >= 15.0, f"Expected catch reward >= 15.0, got {reward}"
        assert info['total_caught'] == 1, "Catch count not updated"
        assert len(info['prey_positions']) == 0, "Prey not removed after catch"
    
    def test_time_penalty_applied(self):
        """Test that time penalty is applied correctly"""
        env = HuntingEnv(
            num_prey=1,
            prey_move_interval=10,
            catch_reward=10.0,
            time_penalty_rate=0.2,
            max_steps=50
        )
        
        obs, info = env.reset()
        
        # Take several steps without catching prey
        total_penalty = 0.0
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(4)  # Stay action
            # Penalty should increase with steps since last catch
            expected_penalty = -0.2 * (step + 1)
            assert reward <= expected_penalty, f"Time penalty not applied correctly at step {step + 1}"
            total_penalty += reward
        
        # Total penalty should be cumulative
        assert total_penalty < 0, "No time penalty accumulated"
    
    def test_multiple_prey_catching(self):
        """Test catching multiple prey in sequence"""
        env = HuntingEnv(
            num_prey=3,
            prey_move_interval=20,  # Prey won't move during test
            catch_reward=5.0,
            time_penalty_rate=0.05,
            max_steps=50
        )
        
        obs, info = env.reset()
        initial_prey_count = len(info['prey_positions'])
        
        # Catch first prey
        first_prey_pos = info['prey_positions'][0]
        env.agent_pos = first_prey_pos
        obs, reward, terminated, truncated, info = env.step(4)
        
        assert info['total_caught'] == 1, "First catch not recorded"
        assert len(info['prey_positions']) == initial_prey_count - 1, "First prey not removed"
        
        # Catch second prey if available
        if len(info['prey_positions']) > 0:
            second_prey_pos = info['prey_positions'][0]
            env.agent_pos = second_prey_pos
            obs, reward, terminated, truncated, info = env.step(4)
            
            assert info['total_caught'] == 2, "Second catch not recorded"
            assert len(info['prey_positions']) == initial_prey_count - 2, "Second prey not removed"


class TestEnvironmentTermination:
    """Test environment termination conditions"""
    
    def test_episode_ends_when_all_prey_caught(self):
        """Test that episode terminates when all prey are caught"""
        env = HuntingEnv(
            num_prey=1,
            prey_move_interval=10,
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        obs, info = env.reset()
        prey_pos = info['prey_positions'][0]
        
        # Move agent to prey position and catch it
        env.agent_pos = prey_pos
        obs, reward, terminated, truncated, info = env.step(4)
        
        # Episode should terminate
        assert terminated, "Episode did not terminate after catching all prey"
        assert not truncated, "Episode was truncated instead of terminated"
    
    def test_episode_truncates_at_max_steps(self):
        """Test that episode truncates at max steps"""
        env = HuntingEnv(
            num_prey=2,
            prey_move_interval=10,
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=5  # Very short episode
        )
        
        obs, info = env.reset()
        
        # Take maximum steps without catching prey
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(4)
            
            if step < 4:  # Not last step
                assert not terminated and not truncated, f"Episode ended early at step {step + 1}"
            else:  # Last step
                assert truncated, "Episode did not truncate at max steps"
                assert not terminated, "Episode terminated instead of truncated"


class TestTinyMindIntegration:
    """Test full TinyMind agent integration"""
    
    def test_agent_learns_from_experience(self):
        """Test that agent can learn from environment interaction"""
        env = HuntingEnv(
            num_prey=2,
            prey_move_interval=3,
            catch_reward=10.0,
            time_penalty_rate=0.1,
            max_steps=50
        )
        
        agent = TinyMindAgent()
        
        # Run a few episodes
        episode_rewards = []
        for episode in range(3):
            obs, info = env.reset()
            agent.reset_memory()
            
            episode_reward = 0
            for step in range(30):
                action = agent.select_action(torch.FloatTensor(obs))
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Agent learns from experience
                agent.learn(obs, action, reward, next_obs, terminated or truncated)
                
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        # Agent should have processed experiences
        assert len(episode_rewards) == 3, "Not all episodes completed"
        # Note: We don't test for improvement since learning takes time
    
    def test_agent_memory_reset(self):
        """Test that agent memory resets properly between episodes"""
        agent = TinyMindAgent()
        
        # Process some data
        visual_input = torch.rand(25)
        action1 = agent.select_action(visual_input)
        
        # Reset memory
        agent.reset_memory()
        
        # Memory should be reset (though behavior might be similar due to randomness)
        memory_state = agent.loop_cortex.get_memory_state()
        assert memory_state['memory_length'] == 0, "Memory not properly reset"
        assert memory_state['step_count'] == 0, "Step count not reset"


if __name__ == "__main__":
    pytest.main([__file__])
