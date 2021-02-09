import gym
import socket
import struct
import numpy
import subprocess
import signal
import select

from gym import spaces


class SuperTuxEnv(gym.Env):

    def __init__(self, proc_name, port, render):
        super(SuperTuxEnv).__init__()
        self.inputs_count = 10010
        self.actions_count = 6
        self.port = port
        self.proc_name = proc_name

        obs_array_low = [-325.0, -600.0, 0, 0, 0]
        obs_array_low.extend([0] * 10000)
        obs_array_high = [-325.0, -600.0, 1, 9, 9]
        obs_array_high.extend([9] * 10000)

        self.observation_space = spaces.Box(low=numpy.array(obs_array_low),
                                            high=numpy.array(obs_array_high),
                                            dtype=numpy.float32)
        self.action_space = spaces.Discrete(self.actions_count)
        self.sock = None
        args = [self.proc_name, "--port", str(self.port)]
        if not render:
            args.extend(["--renderer", "null"])
        self.subproc = subprocess.Popen(args=args, stdout=subprocess.PIPE)

        # Statistics
        self.play_time = .0
        self.player_run_record = .0
        self.player_env_record = .0
        self.victories = 0
        self.episodes = 0
        self.visited_states = []

        # Observations
        self.player_pos_x = .0
        self.player_pos_y = .0
        self.player_velocity_x = .0
        self.player_velocity_y = .0
        self.player_dead = .0
        self.player_won = .0
        self.player_on_ground = .0
        self.player_box_sensors = []
        self.player_ray_sensors = {
            "ground": .0,
            "ground_front": 0
        }

    def close(self):
        if self.sock is not None:
            self.sock.close()
        if self.subproc is not None:
            self.subproc.send_signal(signal.SIGINT)

    def step(self, action):
        old_player_pos_x = self.player_pos_x

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
        # Save visited state
        self.visited_states.append([int(self.play_time), int(self.player_pos_x), int(self.player_pos_y),
                                    int(self.player_velocity_x), int(self.player_velocity_y)])
        done = False
        for sensor_name in self.player_ray_sensors:
            sensor_read = self.player_ray_sensors[sensor_name]
            self.player_ray_sensors[sensor_name] = 2 if sensor_read == 2 or sensor_read == 0 else 1

        # Calculate reward
        reward = .0
        if self.player_ray_sensors["ground_front"] == 2:
            reward += 5 if action == 3 else -5
        reward += 1 if self.player_pos_x > old_player_pos_x else 0
        reward += 2 if self.player_pos_x > self.player_run_record else -1
        if self.play_time >= 120.0 or self.player_dead:
            done = True
            reward -= 1000
        elif self.player_won:
            done = True
            reward += 1000
            self.victories += 1

        # Set player record
        self.player_run_record = max(self.player_pos_x, self.player_run_record)

        info = {}

        if done:
            self.episodes += 1
            self.player_env_record = max(self.player_env_record, self.player_run_record)
            # print(f"I'm done on {self.port}")
            # print(f"My record: {self.player_run_record}")
            # print(f"Environment record: {self.player_env_record}")
            # print(f"Dead: {self.player_dead}")

        return obs, reward, done, info

    def reset(self):
        self.player_run_record = 0
        if self.player_won:
            print(f"Player won! Total victories: {self.victories}")
        if self.sock is not None:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", self.port))

        return self.get_observation()

    def render(self, mode='console'):
        pass

    def receive_message(self):
        chunks = []
        bytes_received = 0
        message_len = self.inputs_count * 4
        while bytes_received < message_len:
            is_ready = select.select([self.sock], [], [], 2)
            if is_ready[0]:
                chunk = self.sock.recv(message_len - bytes_received)
            else:
                self.sock.close()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(("localhost", self.port))
                return self.receive_message()
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_received = bytes_received + len(chunk)
        return b''.join(chunks)

    def get_observation(self):
        self.sock.setblocking(False)

        read_buffer = self.receive_message()

        if len(read_buffer) > 0:
            self.play_time = struct.unpack_from("f", read_buffer, 0)[0]
            self.player_pos_x = struct.unpack_from("f", read_buffer, 4)[0]
            self.player_pos_y = struct.unpack_from("f", read_buffer, 8)[0]
            self.player_velocity_x = struct.unpack_from("f", read_buffer, 12)[0]
            self.player_velocity_y = struct.unpack_from("f", read_buffer, 16)[0]
            self.player_dead = struct.unpack_from("f", read_buffer, 20)[0]
            self.player_won = struct.unpack_from("f", read_buffer, 24)[0]
            self.player_on_ground = struct.unpack_from("f", read_buffer, 28)[0]
            self.player_ray_sensors["ground"] = struct.unpack_from("f", read_buffer, 32)[0]
            self.player_ray_sensors["ground_front"] = struct.unpack_from("f", read_buffer, 36)[0]

            self.player_box_sensors.clear()
            for i in range(10000):
                self.player_box_sensors.append(struct.unpack_from("f", read_buffer, 40 + (i * 4))[0])

        obs = [self.player_velocity_x, self.player_velocity_y, self.player_on_ground,
               self.player_ray_sensors["ground"], self.player_ray_sensors["ground_front"]]
        obs.extend(self.player_box_sensors)

        return numpy.array(obs).astype(numpy.float32)

    def get_visited_states(self):
        state_tuples = []
        for state in self.visited_states:
            state_tuples.append(tuple(state))
        return set(tuple(state_tuples))
