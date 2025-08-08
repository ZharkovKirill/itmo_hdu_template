import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
from half_cheetah_v import HalfCheetahEnvModidified
 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
 
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


cheat_env = HalfCheetahEnvModidified(
    xml_file="/home/kirill/prj/itmo_hdu_template/assets/half_cheetah_modified.xml"
)


eval_callback = EvalCallback(cheat_env,
                            log_path="./ppo_eval/", eval_freq=5000, best_model_save_path="./best_model",
                            n_eval_episodes=2, deterministic=True,
                            render=False)
 
model = PPO("MlpPolicy", cheat_env, verbose=1,  tensorboard_log="log")

model.learn(total_timesteps=int(1e5), progress_bar=True, callback=eval_callback)
model.save("PPO_cheetah")

cheat_env.render_mode = "human"
evaluate_policy(model, cheat_env, render=True, n_eval_episodes=1)
