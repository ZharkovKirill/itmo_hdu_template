"""
PPO Training Script with EvalCallback for HalfCheetah Environment
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gymnasium.wrappers import TimeLimit
from half_cheetah_v import HalfCheetahEnvModidified
import datetime


def make_env(xml_file: str, rank: int = 0, seed: int = 0, log_dir: str = None):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = HalfCheetahEnvModidified(xml_file=xml_file)
        # Add TimeLimit wrapper to ensure episodes don't run forever
        env = TimeLimit(env, max_episode_steps=1000)
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def create_log_dirs(experiment_name: str):
    """Create necessary directories for logging and saving models."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{experiment_name}_{timestamp}"
    
    log_dir = os.path.join(base_dir, "logs")
    eval_log_dir = os.path.join(base_dir, "eval_logs")
    best_model_dir = os.path.join(base_dir, "best_model")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    for directory in [log_dir, eval_log_dir, best_model_dir, checkpoint_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return log_dir, eval_log_dir, best_model_dir, checkpoint_dir


def train_ppo(
    xml_file: str = "assets/half_cheetah_modified.xml",
    total_timesteps: int = 500000,
    n_envs: int = 4,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    save_freq: int = 50000,
    experiment_name: str = "ppo_halfcheetah",
    seed: int = 42
):
    """
    Train PPO agent on HalfCheetah environment with evaluation callback.
    
    Args:
        xml_file: Path to the MuJoCo XML file
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        eval_freq: Frequency of evaluation (in timesteps)
        n_eval_episodes: Number of episodes for evaluation
        save_freq: Frequency of saving checkpoints
        experiment_name: Name for the experiment
        seed: Random seed
    """
    
    # Set random seed
    set_random_seed(seed)
    
    # Create log directories
    log_dir, eval_log_dir, best_model_dir, checkpoint_dir = create_log_dirs(experiment_name)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Best model will be saved to: {best_model_dir}")
    
    # Create vectorized training environment with Monitor wrapper for logging
    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(xml_file, i, seed, log_dir) for i in range(n_envs)])
    else:
        env = HalfCheetahEnvModidified(xml_file=xml_file)
        env = TimeLimit(env, max_episode_steps=5000)
        env = Monitor(env, log_dir)
        train_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = HalfCheetahEnvModidified(xml_file=xml_file)
    # Add TimeLimit wrapper to ensure episodes don't run forever
    eval_env = TimeLimit(eval_env, max_episode_steps=5000)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed
    )
    
    # Create callbacks - Remove TensorboardCallback as PPO already logs rewards
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False  # Disable warnings about env type differences
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name="ppo_run",
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(best_model_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    eval_env.render_mode = "human"
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=5, 
        render=True, 
        deterministic=True
    )
    
    print(f"Final evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model, best_model_dir


def evaluate_trained_model(model_path: str, xml_file: str, n_episodes: int = 5):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model
        xml_file: Path to the MuJoCo XML file
        n_episodes: Number of episodes to evaluate
    """
    # Load the model
    model = PPO.load(model_path)
    
    # Create environment
    env = HalfCheetahEnvModidified(xml_file=xml_file)
    env.render_mode = "human"
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_episodes, 
        render=True, 
        deterministic=True
    )
    
    print(f"Evaluation results - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    # Default configuration
    xml_file = "/home/kirill/prj/itmo_hdu_template/assets/half_cheetah_modified.xml"
    total_timesteps = 500000
    n_envs = 4
    eval_freq = 10000
    n_eval_episodes = 10
    experiment_name = "ppo_halfcheetah"
    seed = 42
    
    # Train new model
    model, model_dir = train_ppo(
        xml_file=xml_file,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        experiment_name=experiment_name,
        seed=seed
    )
    
    print(f"\nTraining completed! Best model saved in: {model_dir}")
    print(f"To evaluate the model, modify the script to call evaluate_trained_model()")
    print(f"\nTo view training progress in TensorBoard, run:")
    print(f"tensorboard --logdir {model_dir}/../logs")
