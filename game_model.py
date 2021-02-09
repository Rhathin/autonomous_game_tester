import sys
import os
import random
import uuid
import numpy
import copy

from dnn_policy import DNNPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
from tensorflow import compat


class GameModel:

    def __init__(self, game_environment, port_numbers=None):
        # Suppress Tensorflow warnings
        compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

        self.uuid = uuid.uuid4()
        self.model_path = f"tmp/model_{str(self.uuid)}"

        # Parse arguments
        self.game_environment = game_environment
        self.proc_name, self.num_envs, self.port_numbers, self.load_model, self.load_path, self.train_model,\
        self.tensorboard_logs_path, self.render, self.steps, self.eval_freq = GameModel.parse_arguments(sys.argv)

        assert self.num_envs is not 0, "--envs: Number of environments cannot be 0"

        # Generate random port numbers for environments to use
        if port_numbers is None:
            while len(self.port_numbers) != self.num_envs + 1:
                GameModel.random_port_numbers(self.port_numbers, self.num_envs + 1)
                self.port_numbers = list(set(self.port_numbers))
        else:
            self.port_numbers.extend(port_numbers)

        # Prepare environment and model
        self.env = VecNormalize(
            DummyVecEnv([GameModel.make_env(game_environment=self.game_environment, proc_name=self.proc_name, port=self.port_numbers[i], render=self.render) for i in range(self.num_envs)]),
            norm_reward=True)
        self.eval_env = VecNormalize(
            DummyVecEnv([GameModel.make_env(game_environment=self.game_environment, proc_name=self.proc_name,
                                            port=self.port_numbers[self.num_envs], render=self.render)]),
            norm_reward=False)

        self.model = None

        if not self.load_model:
            self.model = A2C(DNNPolicy, self.env, tensorboard_log=self.tensorboard_logs_path,
                             lr_schedule="linear", ent_coef=0.0)
        else:
            self.model = A2C.load(self.load_path, self.env, tensorboard_log=self.tensorboard_logs_path,
                             lr_schedule="linear", ent_coef=0.0)

        # Assert that model has been either created or loaded
        assert self.model is not None

    def close(self):
        self.env.close()
        self.eval_env.close()

    @staticmethod
    def make_env(game_environment, proc_name, port, render):
        def _init():
            env = game_environment(proc_name, port, render)
            return env

        return _init

    @staticmethod
    def random_port_numbers(port_numbers, num_envs):
        i = len(port_numbers)
        while i < num_envs:
            port_numbers.append(random.randint(49152, 65535))
            i += 1

    @staticmethod
    def parse_arguments(argv):
        if len(sys.argv) < 2:
            print("You have to specify target game executable name")
            exit(1)

        proc_name = sys.argv[1]
        num_envs = 1
        port_numbers = list()
        load_model = False
        load_path = None
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
                    load_path = str(argv[i])

            if argv[i] == "--no-train":
                train_model = False

            if argv[i] == "--tensorboard-log":
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

        return proc_name, num_envs, port_numbers, load_model, load_path, train_model, tensorboard_logs_path, render, \
               steps, eval_freq

    @staticmethod
    def convert_params_to_floats(params):
        params_floats = list()
        for layer in params:
            if isinstance(params[layer][0], numpy.ndarray):
                for neuron in params[layer]:
                    params_floats.extend(neuron)
            else:
                params_floats.extend(params[layer])

        return params_floats

    @staticmethod
    def convert_floats_to_params(params_floats, params):
        params_copy = copy.deepcopy(params)
        params_floats_index = 0
        for layer in params_copy:
            if isinstance(params_copy[layer][0], numpy.ndarray):
                weights_per_neuron = len(params_copy[layer][0])
                for neuron in params_copy[layer]:
                    neuron[:] = params_floats[params_floats_index:params_floats_index + weights_per_neuron]
                    params_floats_index += weights_per_neuron
            else:
                biases_count = len(params_copy[layer])
                params_copy[layer][:] = params_floats[params_floats_index:params_floats_index + biases_count]
                params_floats_index += biases_count

        return params_copy

    def train(self):
        # Train model
        if self.train_model:
            eval_callback = EvalCallback(self.eval_env, best_model_save_path=f"./models/{self.uuid}/",
                                         log_path=f"./logs/{self.uuid}/", eval_freq=self.eval_freq,
                                         deterministic=True, render=False) if self.eval_freq > 0 else None
            self.model.learn(total_timesteps=self.steps, callback=eval_callback)

            # Save number of unique visited states to log file
            try:
                os.mkdir("logs")
            except FileExistsError:
                pass
            with open(f"./logs/{self.uuid}_states.txt", "a") as file:
                env_idx = 0
                for env in self.env.envs:
                    file.write(f"{self.uuid}-{env_idx}\t{len(env.get_visited_states())}\n")
                    env_idx += 1

    def test(self):
        # Test model
        # Print testing settings
        print("-----------------------------------------")
        print(f"Model testing for {self.proc_name} UUID {self.uuid}")
        print(f"Environments: {self.num_envs}")
        print(f"Rendering: {'On' if self.render else 'Off'}")
        print("-----------------------------------------")
        obs = self.env.reset()
        while True:
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)

    def evaluate(self):
        mean_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
        return mean_reward

    def get_params(self):
        self.model.save(self.model_path)
        data, params = A2C._load_from_file(self.model_path)
        return data, params

    def set_params(self, data, params):
        A2C._save_to_file(self.model_path, data=data, params=params)
        self.model = A2C.load(self.model_path, self.env, tensorboard_log=self.tensorboard_logs_path,
                              lr_schedule="linear", ent_coef=0.0)
