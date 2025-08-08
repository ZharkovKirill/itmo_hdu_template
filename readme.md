# Project Environment Libraries

Below are the main libraries used in this environment (see `env.yaml`):

- **stable-baselines3[extra]**: A set of reliable implementations of reinforcement learning algorithms, with extra utilities for training and evaluation.
- **mujoco**: A physics engine for fast and accurate simulation of rigid body dynamics, commonly used in robotics and RL research.
- **gymnasium_robotics**: Robotics environments for reinforcement learning, compatible with OpenAI Gymnasium interface.
- **mypy**: A static type checker for Python, used to ensure type safety in the codebase.

## Python Files

- **cartpole_example.py**: Example implementation of PPO algorithm training on CartPole-v1 environment using vectorized environments for parallel training.
- **half_cheetah_v.py**: Custom modified HalfCheetah environment implementation based on MuJoCo, a 2D robot simulation for reinforcement learning research.
- **mujoco_model_oppener.py**: MuJoCo model viewer utility that loads and displays the HalfCheetah model with interactive visualization and basic control simulation.
- **rl_example.py**: Reinforcement learning training and evaluation script using SAC algorithm on the custom HalfCheetah environment.
- **ppo_training.py**: Comprehensive PPO training script with EvalCallback, checkpointing, and tensorboard logging for the HalfCheetah environment.
- **config.py**: Configuration file containing hyperparameters and settings for PPO training.

See `env.yaml` for full environment details.
