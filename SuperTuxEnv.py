import gym
import socket
import struct
import numpy

from gym import spaces


class SuperTuxEnv(gym.Env):

    def __init__(self):
        super(SuperTuxEnv).__init__()
        self.observations_count = 3
        self.actions_count = 7
        self.port = 5858
        self.observation_space = spaces.Box(low=0.0, high=10000.0, shape=(3, ), dtype=numpy.float32)
        self.action_space = spaces.Discrete(self.actions_count)
        self.sock = None
        self.play_time = 0
        self.old_player_pos_x = 0
        self.player_pos_x = 0
        self.player_pos_y = 0

    def step(self, action):
        action -= 1
        response = bytearray(b"\x00\x00\x00\x00\x00\x00")
        if action >= 0:
            response[action] = 1

        self.old_player_pos_x = self.player_pos_x

        self.sock.send(response)
        obs = self.get_observation()

        reward = 1 if self.old_player_pos_x < self.player_pos_x else 0
        done = True if self.play_time >= 10.0 else False
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", self.port))
        print("Connection with SuperTux host established")
        return self.get_observation()

    def render(self, mode='console'):
        print("play_time: " + str(self.play_time))
        print("player_pos_x: " + str(self.player_pos_x))
        print("player_pos_y: " + str(self.player_pos_y))
        pass

    def get_observation(self):
        read_buffer = self.sock.recv(self.observations_count * 4)

        if len(read_buffer) > 0:
            self.play_time = struct.unpack_from("f", read_buffer, 0)[0]
            self.player_pos_x = struct.unpack_from("f", read_buffer, 4)[0]
            self.player_pos_y = struct.unpack_from("f", read_buffer, 8)[0]

        return numpy.array([self.play_time, self.player_pos_x, self.player_pos_y]).astype(numpy.float32)
