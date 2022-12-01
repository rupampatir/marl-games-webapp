import numpy as np
import random
import pygame
import sys
import math
from copy import copy
from lib import dqn_model
from lib.utils import mkdir

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = -1

WINDOW_LENGTH = 4
DEFAULT_ENV_NAME = "Connect4"
def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board

def drop_piece(board, row, col, piece):
#   print(board, row, col, piece)
  board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

def print_board(board):
	print(np.flip(board, 0))

def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

def evaluate_window(window, piece):
	score = 0
	opp_piece = PLAYER_PIECE
	if piece == PLAYER_PIECE:
		opp_piece = AI_PIECE

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(EMPTY) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(EMPTY) == 2:
		score += 2

	if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
		score -= 4

	return score


def score_position(board, piece):
	score = 0

	## Score center column
	center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	## Score Horizontal
	for r in range(ROW_COUNT):
		row_array = [int(i) for i in list(board[r,:])]
		for c in range(COLUMN_COUNT-3):
			window = row_array[c:c+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score Vertical
	for c in range(COLUMN_COUNT):
		col_array = [int(i) for i in list(board[:,c])]
		for r in range(ROW_COUNT-3):
			window = col_array[r:r+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score posiive sloped diagonal
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	return score

def is_terminal_node(board):
	return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def get_valid_locations(board):
	valid_locations = []
	for col in range(COLUMN_COUNT):
		if is_valid_location(board, col):
			valid_locations.append(col)
	return valid_locations

def get_invalid_locations(board):
	invalid_locations = []
	for col in range(COLUMN_COUNT):
		if not is_valid_location(board, col):
			invalid_locations.append(col)
	return invalid_locations

def draw_board(board):
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):		
			if board[r][c] == PLAYER_PIECE:
				pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
			elif board[r][c] == AI_PIECE: 
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
	pygame.display.update()

#### MINIMAX ####

def minimax(board, depth, alpha, beta, maximizingPlayer):
	valid_locations = get_valid_locations(board)
	is_terminal = is_terminal_node(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, AI_PIECE):
				return (None, 100000000000000)
			elif winning_move(board, PLAYER_PIECE):
				return (None, -10000000000000)
			else: # Game is over, no more valid moves
				return (None, 0)
		else: # Depth is zero
			return (None, score_position(board, AI_PIECE))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, AI_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value

	else: # Minimizing player
		value = math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, PLAYER_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break
		return column, value

#### DQN #####

### REPLAY MEMORY ###

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000
REPLAY_MIN_SIZE = 1000
LEARNING_RATE = 1e-2
SYNC_TARGET_FRAMES = 1000

EPSILON_START = 1.0
EPSILON_FINAL = 0.2
EPSILON_DECAY_FRAMES = 10**5

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

def normalize(x):
    min = 0
    max = 500
    range = max - min

    return 1.0*(x - min) / range

class Agent:
    def __init__(self, exp_buffer):
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.board = create_board()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None    
        prev_state = self.board.copy()

        if np.random.random() < 0.3:
            available_actions = get_valid_locations(self.board)
            action = random.choice(available_actions)
        else:
            state_a =  np.array([prev_state], copy=False)
            state_v = torch.tensor(state_a).type(torch.cuda.FloatTensor).to(device)
            # state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)

            invalid_actions = get_invalid_locations(self.board)
            # action recommendations from policy net
            q_vals_v[:,invalid_actions] = -1000000
            act_v = torch.argmax(q_vals_v, dim=1)
            action = int(act_v.item())

        row = get_next_open_row(self.board, action)
        drop_piece(self.board, row, action, PLAYER_PIECE)
        new_state = self.board.copy()
        reward = normalize(score_position(self.board, PLAYER_PIECE))
        is_done = is_terminal_node(self.board)
        # do step in the environment
        self.total_reward += reward
        # print(prev_state, action, reward, is_done, new_state)
        exp = Experience([prev_state], action, reward, is_done, [new_state])
        self.exp_buffer.append(exp)
        if is_done:
            done_reward = self.total_reward
            self._reset()
            return done_reward, 1
		
		## PLAY SECOND PLAYERS TURN
        # p = random.random()
        action_p2 = 0
        
        # if (p>0.3):
        action_p2, minimax_score = minimax(self.board, 5, -math.inf, math.inf, True)
        # else:
        #     action_p2 = random.randint(0,COLUMN_COUNT-1)
        while not is_valid_location(self.board, action_p2):
            action_p2, minimax_score = minimax(self.board, 5, -math.inf, math.inf, True)
        row = get_next_open_row(self.board, action_p2)
        drop_piece(self.board, row, action_p2, AI_PIECE)

        if is_terminal_node(self.board):
            done_reward = self.total_reward
            if (winning_move(self.board,AI_PIECE)):
                return done_reward, -1
            self._reset()
        return done_reward, 0

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    mkdir('.', 'checkpoints')
    
    print(device)
    net = dqn_model.DQN((ROW_COUNT,COLUMN_COUNT), COLUMN_COUNT).to(device)
    tgt_net = dqn_model.DQN((ROW_COUNT,COLUMN_COUNT), COLUMN_COUNT).to(device)
    
    # net.load_state_dict(torch.load('./checkpoints/' + DEFAULT_ENV_NAME + "-best.dat"))

    # tgt_net.load_state_dict(torch.load('./checkpoints/' + DEFAULT_ENV_NAME + "-best.dat"))

    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
    print(net)

    buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    agent = Agent(buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    total_results = []
    frame_idx = 0
    ts = time.time()
    best_mean_reward = None
    best_mean_wins = 0
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_FRAMES)

        reward, winner = agent.play_step(net, epsilon, device=device)
        
        if reward is not None:
            total_rewards.append(reward)
            total_results.append(winner)
         
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            mean_wins = np.mean(total_results[-100:])
            print("%d: done %d games, mean reward %.3f, mean wins %.3f, eps %.2f, ts %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, mean_wins, epsilon,
                ts
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("time", ts, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("wins_100", mean_wins, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_wins is None or best_mean_wins < mean_wins:
                torch.save(net.state_dict(), './checkpoints/Scenario2' + DEFAULT_ENV_NAME + "-best.dat")
                if best_mean_wins is not None:
                    print("Best mean wins updated %.3f -> %.3f, model saved" % (best_mean_wins, mean_wins))
                best_mean_wins = mean_wins

        if len(buffer) < REPLAY_MIN_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
  
    writer.close()



""" ------- MAIN ------- """
# net = dqn_model.DQN((ROW_COUNT,COLUMN_COUNT), COLUMN_COUNT).to(device)
# net.load_state_dict(torch.load('./checkpoints/' + DEFAULT_ENV_NAME + "-best.dat"))

# board = create_board()
# print_board(board)
# game_over = False
# pygame.init()

# SQUARESIZE = 100

# width = COLUMN_COUNT * SQUARESIZE
# height = (ROW_COUNT+1) * SQUARESIZE

# size = (width, height)

# RADIUS = int(SQUARESIZE/2 - 5)

# screen = pygame.display.set_mode(size)
# draw_board(board)
# pygame.display.update()

# myfont = pygame.font.SysFont("monospace", 75)

# turn = PLAYER

# while not game_over:
#     print(turn)
#     if turn == PLAYER and not game_over:
#         state_a =  np.array([board.copy()], copy=False)
#         state_v = torch.tensor(state_a).type(torch.cuda.FloatTensor).to(device)
#         # state_v = torch.tensor(state_a).to(device)
#         q_vals_v = net(state_v)

#         invalid_actions = get_invalid_locations(board)
#         # action recommendations from policy net
#         q_vals_v[:,invalid_actions] = -1000000
#         act_v = torch.argmax(q_vals_v, dim=1)
#         col = int(act_v.item())
#         # col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

#         if is_valid_location(board, col):
#             #pygame.time.wait(500)
#             row = get_next_open_row(board, col)
#             drop_piece(board, row, col, PLAYER_PIECE)

#             if winning_move(board, PLAYER_PIECE):
#                 label = myfont.render("AI wins!!", 1, YELLOW)
#                 screen.blit(label, (40,10))
#                 game_over = True

#             print_board(board)
#             draw_board(board)

#             turn += 1
#             turn = turn % 2

#     # # Ask for Player 2 Input
#     elif turn == AI and not game_over:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 sys.exit()

#             if event.type == pygame.MOUSEMOTION:
#                 pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
#                 posx = event.pos[0]
#                 if turn == PLAYER:
#                     pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)

#             pygame.display.update()

#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
#                 #print(event.pos)
#                 # Ask for Player 1 Input
#                 if turn == AI:
#                     posx = event.pos[0]
#                     col = int(math.floor(posx/SQUARESIZE))

#                     if is_valid_location(board, col):
#                         row = get_next_open_row(board, col)
#                         drop_piece(board, row, col, AI_PIECE)

#                         if winning_move(board, AI_PIECE):
#                             label = myfont.render("You wins!!", 1, RED)
#                             screen.blit(label, (40,10))
#                             game_over = True

#                         turn += 1
#                         turn = turn % 2

#                         print_board(board)
#                         draw_board(board)

#     if game_over:
#         pygame.time.wait(3000)