from copy import copy
from lib import wrappers
from lib import dqn_model
from lib.utils import mkdir

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import time
import random
import numpy as np
# import collections
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import cv2
import math

paddleWidth = 4
paddleHeight = 16
player1Offset = 20
player2Offset = 180-paddleWidth

# ball
ballWidth = 2
ballHeight = 4
class Pong():

    def __init__(self):
        self.board = []
        self.width = 200
        self.height = 200
        self.player_1_position = 100
        self.player_2_position = 100   
        self.ball_position = [int(self.height/2),int(self.width/2)]
        self.ball_direction_x = random.choice([1,2,-1,-2])
        self.ball_direction_y =random.choice([1,2,-1,-2])
        self.lastFourFrames = []
        self.frame = 0

    def reset(self):
        self.player_1_position = 100
        self.player_2_position = 100   
        self.ball_position = [int(self.height/2),int(self.width/2)]
        self.ball_direction_x = random.choice([1,-1])
        self.ball_direction_y = random.choice([1,-1])
        self.lastFourFrames = []
        self.renderImage()
        return self.board
    
    def is_overlapping(self, rec1, rec2):
        widthIsPositive = min(rec1[2], rec2[2]) > max(rec1[0], rec2[0])
        heightIsPositive = min(rec1[3], rec2[3]) > max(rec1[1], rec2[1])
        return ( widthIsPositive and heightIsPositive)
    

    def renderImage(self):
        board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append([0,0,0])
            board.append(row)
        
        
        # Player 1 paddle
        for i in range(self.player_1_position, self.player_1_position+paddleHeight):#(i=self.player_1_position i<self.player_1_position+paddleHeight i+=1) 
            for j in range(player1Offset, player1Offset+paddleWidth):
                board[i][j] = [255,255,255]
            
        
        
        # Player 2 paddle
        for i in range(self.player_2_position, self.player_2_position+paddleHeight):
            for j in range(player2Offset,player2Offset+paddleWidth):
                board[i][j] = [255,255,255]
    
        
        # console.log(self.ball_position)
        for i in range(self.ball_position[0],self.ball_position[0]+ballHeight):
            for j in range(self.ball_position[1],self.ball_position[1]+ballWidth): 
                board[i][j] = [128,128,128]
                    
        self.board = board

    def _get_distance(self, p1, p2):
      return np.sqrt(np.sum(np.square(p1 - p2)))

    def step(self, actions):      

        # Perform actions
        if actions[0] == 1 and self.player_1_position-5>0:
          self.player_1_position-=5
        if actions[0] == 2 and self.player_1_position+paddleHeight+5<self.height:
          self.player_1_position+=5
        if actions[1] == 1 and self.player_2_position-5>0:
          self.player_2_position-=5
        if actions[1] == 2 and self.player_2_position+paddleHeight+5<self.height:
          self.player_2_position+=5
        # Update ball position
        # if (self.ball_position[0]+paddleHeight<self.height and self.ball_position[0]>0):
        #     self.player_1_position = self.ball_position[0]

        ball_y = self.ball_position[0]
        ball_x = self.ball_position[1]
        # if it hit the paddle on left
        
        # console.log(ball_y+ballHeight,paddle_top)
        # [x1, y1, x2, y2], where (x1, y1) is the coordinates of its bottom-left corner, and (x2, y2) is the coordinates of its top-right corner. 
        ball_rect = [ball_x, self.height-(ball_y+ballHeight), ball_x+ballWidth, self.height-ball_y]
        paddle_rect = [player1Offset, self.height-(self.player_1_position+paddleHeight), player1Offset+paddleWidth, self.height-self.player_1_position]
        
        if (self.is_overlapping(ball_rect,paddle_rect)):
            self.ball_direction_x=self.ball_direction_x*-1
        # if it hit the paddle on right
        paddle_rect = [player2Offset, self.height-(self.player_2_position+paddleHeight), player2Offset+paddleWidth, self.height-self.player_2_position]
        if (self.is_overlapping(ball_rect,paddle_rect)):
            self.ball_direction_x=self.ball_direction_x*-1

        # if it hit top screen
        if (ball_y<=0):
            self.ball_direction_y=self.ball_direction_y*-1
        
        # if it hit the bottom screen
        if (ball_y+ballHeight>=self.height-1):
            self.ball_direction_y=self.ball_direction_y*-1
        done = False

        # if it hit left or right screens
        if (ball_x<=0 or ball_x+ballWidth>=self.width-5):
            # self.gameOver()
            done = True

        self.ball_position = [self.ball_position[0] + self.ball_direction_y, self.ball_position[1] + self.ball_direction_x]

        # if (self.lastFourFrames.length>=4):
        #     if (self.frame%4==0):
        #         self.lastFourFrames.popleft()
        #         self.lastFourFrames.append(board.copy())
        #     self.frame+=1
        # else:
        #     self.lastFourFrames.append(board.copy()) 

        # Get rewards
        rewards = [-self._get_distance(ball_y, paddle) for paddle in [self.player_1_position, self.player_2_position]]
        if (ball_x<=0):
            rewards[0] = -1000000
            rewards[1] = 1000000
        elif (ball_x+ballWidth>=self.width):
            rewards[1] = -1000000
            rewards[0] = 1000000
        self.renderImage()
        return self.board.copy(), rewards, done, None
        
    def render(self):
        plt.imshow(self.board)
        plt.show(block=False)
        plt.pause(0.004)
        plt.close()




DEFAULT_ENV_NAME = "PONG"

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
REPLAY_MIN_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_FRAMES = 10**4


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.env.reset()
        self.state, _, _, _ = self.play_four_steps([0,0])
        self.total_reward = 0.0

    def process_pong_frame(self, observations):

        final = []
        # Resize image
        for frame in observations:
            frame = np.array(frame).astype('float32')
            frame = cv2.resize(np.array(frame), (84,84), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            final.append(np.array(frame).astype(np.float32) / 255.0)

        return np.array(final)


    def play_four_steps(self, actions):
        frame1, reward1, done1, _ = self.env.step(actions)
        frame2, reward2, done2, _ = self.env.step(actions)
        frame3, reward3, done3, _ = self.env.step(actions)
        frame4, reward4, done4, _ = self.env.step(actions)

        state = self.process_pong_frame([frame1, frame2, frame3,frame4])
        reward = [reward1[0] + reward2[0] + reward3[0] + reward4[0], reward1[1] + reward2[1] + reward3[1] + reward4[1]]
        done = done1 or done2 or done3 or done4
        
        return state, reward, done, None
    def play_step(self, net, epsilon=0.0, device="cpu"):

        # get four frames, with same action
        done_reward = None
        
        # Two actions, flip
        
        action1, action2 = random.choice([0,1,2]),random.choice([0,1,2])

        state = self.state
        flipped_state = np.flip(state, axis=2)
        if np.random.random() > epsilon:
            # player 1 state
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action1 = int(act_v.item())
            
            state_a = np.array([flipped_state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action2 = int(act_v.item())
            print(action1, action2)
        # do step in the environment
        new_state, rewards, is_done, _ = self.play_four_steps([action1, action2])
        self.total_reward += sum(rewards)
        new_state = new_state
        exp = Experience(state, action1, rewards[0], is_done, new_state)
        # exp = Experience(flipped_state, action2, rewards[1], is_done, new_state)

        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
   
    states_v = torch.tensor(states, dtype=torch.float).to(device)
    
   
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


# if __name__ == "__main__":
#     device = "cpu"
#     env = Pong()
#     net = dqn_model.DQN((4,84,84), 3).to(device)
#     tgt_net = dqn_model.DQN((4,84,84), 3).to(device)
#     writer = SummaryWriter(comment="-" + "PONG")
#     buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
#     mkdir('.', 'checkpointsnew')
#     agent = Agent(env, buffer)
#     epsilon = EPSILON_START
#     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
#     total_rewards = []
#     frame_idx = 0
#     ts_frame = 0
#     ts = time.time()
#     best_mean_reward = None

#     while True:
#         frame_idx += 1
#         epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_FRAMES)
#         env.render()
#         reward = agent.play_step(net, epsilon, device=device)
#         if reward is not None:
#             total_rewards.append(reward)
#             speed = (frame_idx - ts_frame) / (time.time() - ts)
#             ts_frame = frame_idx
#             ts = time.time()
#             mean_reward = np.mean(total_rewards[-100:])
#             print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
#                 frame_idx, len(total_rewards), mean_reward, epsilon,
#                 speed
#             ))
#             writer.add_scalar("epsilon", epsilon, frame_idx)
#             writer.add_scalar("speed", speed, frame_idx)
#             writer.add_scalar("reward_100", mean_reward, frame_idx)
#             writer.add_scalar("reward", reward, frame_idx)
#             if best_mean_reward is None or best_mean_reward < mean_reward:
#                 torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
#                 if best_mean_reward is not None:
#                     print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
#                 best_mean_reward = mean_reward

#         if len(buffer) < REPLAY_MIN_SIZE:
#             continue

#         if frame_idx % SYNC_TARGET_FRAMES == 0:
#             tgt_net.load_state_dict(net.state_dict())

#         optimizer.zero_grad()
#         batch = buffer.sample(BATCH_SIZE)
#         loss_t = calc_loss(batch, net, tgt_net, device=device)
#         loss_t.backward()
#         optimizer.step()
  
#     writer.close()




if __name__ == "__main__":
    # mkdir('.', 'checkpoints')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    # parser.add_argument("--env", default=DEFAULT_ENV_NAME,
    #                     help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    # parser.add_argument("--reward", type=float, default=MEAN_REWARD_GOAL,
    #                     help="Mean reward goal to stop training, default=%.2f" % MEAN_REWARD_GOAL)
    # args = parser.parse_args()
    device = "cpu"

    env = Pong()
    net = dqn_model.DQN((4,84,84), 3).to(device)
    net.load_state_dict(torch.load(DEFAULT_ENV_NAME + "-best.dat", map_location=torch.device('cpu')))

   
    tgt_net = dqn_model.DQN((4,84,84), 3).to(device)
    # writer = SummaryWriter(comment="-" + args.env)
    # print(net)

    buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    
    # def saveImage(image, image_name):
    #     print(image.shape)
    #     im = Image.fromarray(image)
    #     im.save(image_name)
    

    agent = Agent(env, buffer)

    # state = env.reset()
    while True:
        agent.play_step(net, 0, device=device)
        agent.env.render()
        # time.sleep(0.04)

         
