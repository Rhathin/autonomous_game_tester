import sys
import random

from dnn_policy import DnnPolicy
from stable_baselines import A2C
from super_tux_env import SuperTuxEnv
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback
from tensorflow import compat


def make_env(proc_name, port, render):
    def _init():
        env = SuperTuxEnv(proc_name, port, render)
        return env
    return _init


def random_port_numbers(port_numbers, num_envs):
    i = len(port_numbers)
    while i < num_envs:
        port_numbers.append(random.randint(49152, 65535))
        i += 1


def parse_arguments(argv):
    if len(sys.argv) < 2:
        print("You have to specify target game executable name")
        exit(1)

    proc_name = sys.argv[1]
    num_envs = 1
    port_numbers = list()
    load_model = False
    model_path = None
    train_model = True
    tensorboard_logs_path = None
    render = True
    steps = 1000000
    eval_freq = 50000

    for i in range(len(argv)):
        if argv[i] == "--envs":
            i += 1
            if i > len(argv):
                print("Need to specify number of environments for --envs")
            else:
                num_envs = int(argv[i])
        
        if argv[i] == "--load":
            i += 1
            if i > len(argv):
                print("Need to specify saved model path for --load")
            else:
                load_model = True
                model_path = str(argv[i])
            
        if argv[i] == "--no-train":
            train_model = False

        if argv[i] == "--tensorboard_log":
            i += 1
            if i > len(argv):
                print("Need to specify directory to save Tensorboard logs for --tensorboard_log")
            else:
                tensorboard_logs_path = str(argv[i])

        if argv[i] == "--no-render":
            render = False

        if argv[i] == "--steps":
            i += 1
            if i > len(argv):
                print("Need to specify steps number for --steps")
            else:
                steps = int(argv[i])

        if argv[i] == "--eval-freq":
            i += 1
            if i > len(argv):
                print("Need to specify evaluation frequency for --eval_freq")
            else:
                eval_freq = int(argv[i])
    
    return proc_name, num_envs, port_numbers, load_model, model_path, train_model, tensorboard_logs_path, render,\
           steps, eval_freq


def main():
    # Suppress Tensorflow warnings
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    # Parse arguments
    proc_name, num_envs, port_numbers, load_model, model_path, train_model, tensorboard_logs_path, render, steps, eval_freq \
        = parse_arguments(sys.argv)

    assert num_envs is not 0, "--envs: Number of environments cannot be 0"

    # Generate random port numbers for environments to use
    while len(port_numbers) != num_envs + 1:
        random_port_numbers(port_numbers, num_envs + 1)
        port_numbers = list(set(port_numbers))

    # Prepare environment and model
    env = VecNormalize(DummyVecEnv([make_env(proc_name=proc_name, port=port_numbers[i], render=render) for i in range(num_envs)]),
                       norm_reward=False)

    model = None

    if not load_model:
        model = A2C(DnnPolicy, env, tensorboard_log=tensorboard_logs_path,
                    lr_schedule="linear", ent_coef=0.0)
    else:
        model = A2C.load(model_path, env, tensorboard_log=tensorboard_logs_path,
                    lr_schedule="linear", ent_coef=0.0)

    # Assert that model has been either created or loaded
    assert model is not None

    # Train model
    if train_model:

        # Print training settings
        print("-----------------------------------------")
        print(f"Model training for {proc_name}")
        print(f"Total timesteps: {steps}")
        print(f"Evaluation Frequency: {eval_freq}")
        print(f"Environments: {num_envs}")
        print(f"Rendering: {'On' if render else 'Off'}")
        print("-----------------------------------------")

        eval_env = VecNormalize(DummyVecEnv([make_env(proc_name=proc_name, port=port_numbers[num_envs], render=render)]),
                                norm_reward=False) if eval_freq > 0 else None
        eval_callback = EvalCallback(eval_env, best_model_save_path='./models/',
                                     log_path='./logs/', eval_freq=eval_freq,
                                     deterministic=True, render=False) if eval_freq > 0 else None
        model.learn(total_timesteps=steps, callback=eval_callback)

    # Test model
    # Print testing settings
    print("-----------------------------------------")
    print(f"Model testing for {proc_name}")
    print(f"Environments: {num_envs}")
    print(f"Rendering: {'On' if render else 'Off'}")
    print("-----------------------------------------")

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)


main()
