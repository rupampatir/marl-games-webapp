# Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import cv2
import gym
import gym.spaces
import numpy as np
import collections
import matplotlib.pyplot as plt
from PIL import Image
j = 0
def saveImage(image, image_name):
    im = Image.fromarray(image)
    im.save(image_name)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user needs to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        i = 0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            saveImage(obs,"imagesaved"+str(i)+".png")
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
            i+=1
        global j
        j+=1
        if (j==30):
            exit()
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        # print(max_frame.shape)
        seta = set()
        for row in max_frame:
            for col in row:
                seta.add(tuple([col[0],col[1],col[2]]))
        print("Before:", seta)
        
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        img = np.reshape(frame, frame.shape).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        observation = np.moveaxis(observation, 2, 0)
        
        return observation

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        print("After:", np.unique(np.array(obs).astype(np.float32) / 255.0))
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def make_env(env_name):
    env = gym.make(env_name)
    state = env.reset()

    env = MaxAndSkipEnv(env)
    # plt.imsave("2.png", env.render(mode='rgb_array'))

    env = FireResetEnv(env)
    # plt.imsave("3.png", env.render(mode='rgb_array'))

    env = ProcessFrame(env)
    # plt.imsave("4.png", env.render(mode='rgb_array'))

    env = ImageToPyTorch(env)
    # plt.imsave("5.png", env.render(mode='rgb_array'))

    env = BufferWrapper(env, 4)
    # plt.imsave("6.png", env.render(mode='rgb_array'))

    env = ScaledFloatFrame(env)
    # plt.imsave("7.png", env.render(mode='rgb_array'))
    
    return env



def process_frontend_frame(observations):
    resized_obs = []
    # Resize image
    for frame in observations:
        img = np.reshape(frame, frame.shape).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        resized_obs.append(x_t.astype(np.uint8))

    resized_obs = np.array()

    # Move channel to front
    observation =  np.moveaxis(observation, 2, 0)


    observation = np.array(observation).astype(np.float32) / 255.0

    