"""
TinyMind CLI: Simple command-line interface

Minimal CLI for TinyMind using industry-standard libraries.
"""

import click
import numpy as np
from pathlib import Path

from .agent import TinyMindAgent
from .envs.hunting_env import make_tinymind_env


@click.group()
@click.version_option()
def cli():
    """🧠 TinyMind: Biologically-inspired AI"""
    pass


@cli.command()
@click.option("--episodes", default=10, help="Number of episodes")
@click.option("--env", default="empty", help="Environment type")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed learning analysis")
def demo(episodes, env, verbose):
    """Run a simple demo with SARSA learning"""
    click.echo(f"🧠 TinyMind Demo - {episodes} episodes")
    if verbose:
        click.echo("📊 Verbose mode: Detailed learning analysis enabled")
    
    # Create agent and environment
    agent = TinyMindAgent()
    environment = make_tinymind_env(env_name=env.title())
    
    all_rewards = []
    all_steps = []
    
    for episode in range(episodes):
        obs, _ = environment.reset()
        agent.reset_memory()
        total_reward = 0
        steps = 0
        
        # Get visual field size from config
        from .config import ENVIRONMENT
        visual_h, visual_w = ENVIRONMENT["visual_field_size"]
        
        # First action without reward
        action = agent.act(obs.reshape(visual_h, visual_w), reward=0.0, done=False)
        
        for step in range(ENVIRONMENT["max_steps"]):
            obs, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            steps += 1
            
            # Next action with reward (SARSA learning)
            action = agent.act(obs.reshape(visual_h, visual_w), reward=reward, done=(terminated or truncated))
            
            if terminated or truncated:
                break
        
        all_rewards.append(total_reward)
        all_steps.append(steps)
        
        if verbose:
            # Show detailed analysis every few episodes
            if episode % max(1, episodes // 5) == 0 or episode == episodes - 1:
                info = agent.get_info()
                learning_progress = info.get('learning_progress', {})
                click.echo(f"\n📈 Episode {episode+1} Analysis:")
                click.echo(f"   Steps: {steps}, Total Reward: {total_reward:.3f}")
                click.echo(f"   Epsilon: {learning_progress.get('epsilon', 'N/A')}")
                click.echo(f"   Avg Reward/Step: {learning_progress.get('avg_reward_per_step', 'N/A')}")
                
                # Show cortex statistics
                rl_stats = info.get('rl_cortex', {})
                visual_stats = info.get('visual_cortex', {})
                loop_stats = info.get('loop_cortex', {})
                
                # Q-value statistics
                q_stats = rl_stats.get('q_value_stats', {})
                if q_stats:
                    click.echo(f"   Q-values: max={q_stats.get('max_q', 0):.3f}, "
                             f"avg={q_stats.get('avg_q', 0):.3f}, "
                             f"min={q_stats.get('min_q', 0):.3f}")
                
                # Visual cortex learning statistics
                if visual_stats and 'training_loss' in visual_stats:
                    click.echo(f"   VisualCortex: loss={visual_stats.get('training_loss', 0):.4f}, "
                             f"lr={visual_stats.get('learning_rate', 0):.4f}")
                
                # Loop cortex learning statistics  
                if loop_stats and 'training_loss' in loop_stats:
                    click.echo(f"   LoopCortex: loss={loop_stats.get('training_loss', 0):.4f}, "
                             f"memory_size={loop_stats.get('memory_size', 0)}")
                
                # Show recent performance trend
                if len(all_rewards) >= 3:
                    recent_avg = np.mean(all_rewards[-3:])
                    early_avg = np.mean(all_rewards[:3]) if len(all_rewards) > 3 else recent_avg
                    improvement = recent_avg - early_avg
                    click.echo(f"   Learning Trend: {improvement:+.3f} (recent vs early)")
        else:
            click.echo(f"Episode {episode+1}: {steps} steps, reward: {total_reward:.3f}")
    
    # Final summary
    if len(all_rewards) > 1:
        click.echo(f"\n📊 Final Summary:")
        click.echo(f"   Average reward: {np.mean(all_rewards):.3f}")
        click.echo(f"   Best reward: {np.max(all_rewards):.3f}")
        click.echo(f"   Average steps: {np.mean(all_steps):.1f}")
        
        # Learning progress analysis
        if len(all_rewards) >= 6:
            early_rewards = all_rewards[:len(all_rewards)//3]
            late_rewards = all_rewards[-len(all_rewards)//3:]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            click.echo(f"   Learning improvement: {improvement:+.3f}")
            
            if improvement > 0.01:
                click.echo("   ✅ Agent is learning successfully!")
            elif improvement > -0.01:
                click.echo("   ⚠️  Agent learning is stable")
            else:
                click.echo("   ❌ Agent may need parameter tuning")
        
        # Final cortex status
        final_info = agent.get_info()
        click.echo(f"\n🧠 Final Cortex Status:")
        
        # Visual Cortex
        visual_info = final_info.get('visual_cortex', {})
        if visual_info:
            click.echo(f"   VisualCortex: {visual_info.get('architecture', 'N/A')} - "
                     f"Loss: {visual_info.get('training_loss', 0):.4f}")
        
        # Loop Cortex
        loop_info = final_info.get('loop_cortex', {})
        if loop_info:
            click.echo(f"   LoopCortex: {loop_info.get('architecture', 'N/A')} - "
                     f"Loss: {loop_info.get('training_loss', 0):.4f}")
        
        # RL Cortex
        rl_info = final_info.get('rl_cortex', {})
        if rl_info:
            q_stats = rl_info.get('q_value_stats', {})
            click.echo(f"   RLCortex: SARSA - "
                     f"ε={rl_info.get('epsilon', 0):.3f}, "
                     f"Q̄={q_stats.get('avg_q', 0):.3f}")
    
    environment.close()


@cli.command()
@click.option("--timesteps", default=1000, help="Training timesteps")
def train(timesteps):
    """Train TinyMind agent"""
    click.echo(f"🎯 Training TinyMind for {timesteps} timesteps")
    
    agent = TinyMindAgent()
    
    # Generate some training data
    from .config import ENVIRONMENT
    visual_h, visual_w = ENVIRONMENT["visual_field_size"]
    training_data = [np.random.random((visual_h, visual_w)) for _ in range(100)]
    sequences = [np.random.random((20, 8)) for _ in range(50)]
    
    # Train cortices
    agent.visual_cortex.train(training_data)
    agent.loop_cortex.train(sequences)
    agent.rl_cortex.train(timesteps)
    
    click.echo("✅ Training completed!")


@cli.command()
@click.option("--config", "-c", is_flag=True, help="Show detailed configuration")
def info(config):
    """Show TinyMind system information"""
    if config:
        # Show detailed configuration
        from .config import get_config_summary, SYSTEM_INFO
        
        click.echo("🧠 TinyMind Configuration Details:")
        click.echo("=" * 50)
        
        summary = get_config_summary()
        for key, value in summary.items():
            click.echo(f"{key:15}: {value}")
        
        click.echo("\n📊 Parameter Breakdown:")
        params = SYSTEM_INFO["total_parameters"]
        for cortex, count in params.items():
            if cortex != "total":
                click.echo(f"  {cortex:12}: {count:,} parameters")
        click.echo(f"  {'Total':12}: {params['total']:,} parameters")
        
        click.echo(f"\n🎯 Key Features:")
        for feature in SYSTEM_INFO["key_features"]:
            click.echo(f"  • {feature}")
    else:
        # Show basic system info
        agent = TinyMindAgent()
        info = agent.get_info()
        
        click.echo("🧠 TinyMind System Information:")
        for cortex_name, cortex_info in info.items():
            if isinstance(cortex_info, dict) and 'algorithm' in cortex_info:
                click.echo(f"  {cortex_name}: {cortex_info['algorithm']}")


def main():
    """Entry point for CLI"""
    cli()


if __name__ == "__main__":
    main() 