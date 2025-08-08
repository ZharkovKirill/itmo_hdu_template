"""
Configuration file for PPO training parameters
"""

# Training Configuration
TRAINING_CONFIG = {
    "total_timesteps": 500000,
    "n_envs": 4,
    "seed": 42,
    "xml_file": "assets/half_cheetah_modified.xml"
}

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
}

# Evaluation Configuration
EVAL_CONFIG = {
    "eval_freq": 10000,
    "n_eval_episodes": 10,
    "save_freq": 50000
}

# Environment Configuration
ENV_CONFIG = {
    "xml_file": "assets/half_cheetah_modified.xml",
    "render_mode": None  # Set to "human" for visualization during training
}
