import gymnasium as gym
from stable_baselines3 import PPO, SAC
from half_cheetah_v import HalfCheetahEnvModidified
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
cheat_env = HalfCheetahEnvModidified()

model = SAC("MlpPolicy", cheat_env, verbose=1)

#model.learn(total_timesteps=int(2e5), progress_bar=True)
#model.save("SAC_cheetah")

model = SAC.load("SAC_cheetah.zip", env=cheat_env)
cheat_env.render_mode = "human"
evaluate_policy(model, cheat_env, render = True, n_eval_episodes=1)