import gym
import socket
import struct
import numpy
import copy
import subprocess
import signal
import select

from gym import spaces


class SuperTuxEnv(gym.Env):

    def __init__(self, proc_name, port, render):
        super(SuperTuxEnv).__init__()
        self.inputs_count = 20
        self.actions_count = 6
        self.outputs_count = 3
        self.port = port
        self.proc_name = proc_name
        self.observation_space = spaces.Box(low=numpy.array([-325.0, -600.0,
                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                            high=numpy.array([325.0, 600.0,
                                                              9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1]),
                                            dtype=numpy.float32)
        self.action_space = spaces.Discrete(self.actions_count)
        self.sock = None
        args = [self.proc_name, "--port", str(self.port)]
        if not render:
            args.extend(["--renderer", "null"])
        self.subproc = subprocess.Popen(args=args)

        # Observations
        self.play_time = .0
        self.player_pos_x = .0
        self.player_velocity_x = .0
        self.player_velocity_y = .0
        self.player_run_record = .0
        self.player_env_record = .0
        self.player_dead = .0
        self.player_on_ground = .0
        self.player_sensors = {
            "x": .0,
            "nx": .0,
            "y": .0,
            "ny": .0,
            "xy": .0,
            "xny": .0,
            "lx": .0,
            "lnx": .0,
            "ly": .0,
            "lny": .0,
            "lxy": .0,
            "lxny": .0,
            "ground": .0,
            "ground_front": .0
        }

    def __del__(self):
        self.subproc.send_signal(signal.SIGINT)

    def step(self, action):
        old_player_sensors = copy.deepcopy(self.player_sensors)

        # Send response corresponding to received action
        # Subtract 1 from action in order to allow model perform no action i.e. wait
        action -= 1
        response = bytearray(b"\x00\x00\x00")
        if 0 <= action < 3:
            response[action] = 1
        elif action == 3:
            response[1] = 1
            response[2] = 1
        elif action == 4:
            response[0] = 1
            response[2] = 1
        self.sock.send(response)
        obs = self.get_observation()

        # Calculate reward
        enemy_nearby = False
        reward = .0
        for sensor in self.player_sensors.items():
            if sensor == 5:
                enemy_nearby = True
                reward -= 1
        # Add acceleration info
        reward += 0.2 if old_player_sensors["x"] == 1 and action == 2 else .0
        reward += 1 if self.player_pos_x > self.player_run_record else .0
        reward += 0.5 if self.player_velocity_x > 160 and self.player_on_ground == 1 and not enemy_nearby else .0
        reward += 5 if self.player_sensors["ground_front"] == 2 and action == 3 else .0
        reward += 5 if self.player_sensors["lxny"] == 5 and self.player_sensors["lx"] != 5 and action == 4 else .0

        reward -= 1000 if self.player_dead >= 1.0 else .0

        # Set player record
        self.player_run_record = max(self.player_pos_x, self.player_run_record)

        done = True if self.play_time >= 120.0 else False
        info = {}

        if self.player_dead:
            done = True

        if done:
            self.player_env_record = max(self.player_env_record, self.player_run_record)
            print(f"I'm done on {self.port}")
            print(f"My record: {self.player_run_record}")
            print(f"Environment record: {self.player_env_record}")
            print(f"Dead: {self.player_dead}")

        return obs, reward, done, info

    def reset(self):
        self.player_run_record = 0
        if self.sock is not None:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", self.port))

        return self.get_observation()

    def render(self, mode='console'):
        pass

    def get_observation(self):
        self.sock.setblocking(False)
        is_ready = select.select([self.sock], [], [], 5)
        if is_ready[0]:
            read_buffer = self.sock.recv(self.inputs_count * 4)
        else:
            print(f"recv timeout on {self.port}")
            print(f"Trying to reconnect on {self.port}...")
            self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(("localhost", self.port))
            return self.get_observation()

        if len(read_buffer) > 0:
            self.play_time = struct.unpack_from("f", read_buffer, 0)[0]
            self.player_pos_x = struct.unpack_from("f", read_buffer, 4)[0]
            self.player_velocity_x = struct.unpack_from("f", read_buffer, 8)[0]
            self.player_velocity_y = struct.unpack_from("f", read_buffer, 12)[0]
            self.player_sensors["x"] = struct.unpack_from("f", read_buffer, 16)[0]
            self.player_sensors["nx"] = struct.unpack_from("f", read_buffer, 20)[0]
            self.player_sensors["y"] = struct.unpack_from("f", read_buffer, 24)[0]
            self.player_sensors["ny"] = struct.unpack_from("f", read_buffer, 28)[0]
            self.player_sensors["xy"] = struct.unpack_from("f", read_buffer, 32)[0]
            self.player_sensors["xny"] = struct.unpack_from("f", read_buffer, 36)[0]
            self.player_sensors["lx"] = struct.unpack_from("f", read_buffer, 40)[0]
            self.player_sensors["lnx"] = struct.unpack_from("f", read_buffer, 44)[0]
            self.player_sensors["ly"] = struct.unpack_from("f", read_buffer, 48)[0]
            self.player_sensors["lny"] = struct.unpack_from("f", read_buffer, 52)[0]
            self.player_sensors["lxy"] = struct.unpack_from("f", read_buffer, 56)[0]
            self.player_sensors["lxny"] = struct.unpack_from("f", read_buffer, 60)[0]
            self.player_sensors["ground"] = struct.unpack_from("f", read_buffer, 64)[0]
            self.player_sensors["ground_front"] = struct.unpack_from("f", read_buffer, 68)[0]
            self.player_dead = struct.unpack_from("f", read_buffer, 72)[0]
            self.player_on_ground = struct.unpack_from("f", read_buffer, 76)[0]

            # Treat holes the same way as spikes
            self.player_sensors["ground"] = 2 if self.player_sensors["ground"] == 0 else self.player_sensors["ground"]
            self.player_sensors["ground_front"] = 2 if self.player_sensors["ground_front"] == 0 else self.player_sensors["ground"]
            # Ignore all collectibles (e.g. Coins)
            for sensor in self.player_sensors:
                if self.player_sensors[sensor] == 8:
                    self.player_sensors[sensor] = 0

        return numpy.array([self.player_velocity_x, self.player_velocity_y, self.player_sensors["x"],
                            self.player_sensors["nx"], self.player_sensors["y"], self.player_sensors["ny"],
                            self.player_sensors["xy"], self.player_sensors["xny"], self.player_sensors["lx"],
                            self.player_sensors["lnx"], self.player_sensors["ly"], self.player_sensors["lny"],
                            self.player_sensors["lxy"], self.player_sensors["lxny"], self.player_sensors["ground"],
                            self.player_sensors["ground_front"], self.player_on_ground]).astype(numpy.float32)
