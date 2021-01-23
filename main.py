from DnnPolicy import DnnPolicy
from stable_baselines import A2C
from SuperTuxEnv import SuperTuxEnv
from stable_baselines.common.env_checker import check_env

# parallel environments
env = SuperTuxEnv()
model = A2C(DnnPolicy, env)
# check_env(env)
# Train model
print("Model train")
model.learn(total_timesteps=1000)
model.save("supertux_dnna2c")
model = A2C.load("supertux_dnna2c", policy=DnnPolicy)

print("Model start")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
