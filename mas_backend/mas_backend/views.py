from django.shortcuts import render
import os
import numpy as np
import random as random
import pickle
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import cv2
import torch
import math

## TIC TAC TOE
policy_1_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'policy_Q_1')
policy_1 = pickle.load(open(policy_1_path, "rb"))
policy_2_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'policy_Q_2_both')
policy_2 = pickle.load(open(policy_2_path, "rb"))


def home(request):    
    return render(request, 'index.html')

def get_available_squares(board):
    available_squares = []
    for row in range(3):
      for column in range(3):
        if board[row][column] == 0:
          available_squares.append((row,column))
    return available_squares

def get_hash(board):
    board_hash = str(board.astype(int).reshape(3 * 3))
    return board_hash

def chooseAction_QTable(current_board, start_player):
        policy = None
        if (start_player==2):
            policy = policy_1
        else:
            policy = policy_2
        exp_rate = 0.3
        positions = get_available_squares(current_board)
        current_state = get_hash(current_board)
        best_action = []
        best_action_value = -np.Inf
        for action in positions:
            Q_s_a = policy[current_state][action]
            if Q_s_a == best_action_value:
                best_action.append(action)
            elif Q_s_a > best_action_value:
                best_action = [action]
                best_action_value = Q_s_a
        best_action = random.choice(best_action)
        
        n_actions =len(positions)
        p = np.full(n_actions,exp_rate/n_actions)
        p[positions.index(best_action)]+= 1 - exp_rate
      
        return positions[np.random.choice(len(positions),p=p)]
        

# our result page view
@csrf_exempt
def get_tic_tac_toe_action(request):
    data = json.loads(request.body)
    board = data['board']
    start_player = data['start_player']
    optimalAction = chooseAction_QTable(np.array(board), start_player)
    return JsonResponse({"action":optimalAction}, safe=False)


### PONG

import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# PING PONG
net = DQN((4,84,84), 6).to("cpu")
DEFAULT_ENV_NAME = "./PongNoFrameskip-v4"
net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DEFAULT_ENV_NAME + "-best.dat"), map_location=torch.device('cpu')))


def process_pong_frame(observations):

    resized_obs = []
    # Resize image
    for frame in observations:
        img = np.reshape(frame, frame.shape).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]

        x_t = np.reshape(x_t, [84, 84])
        resized_obs.append(x_t.astype(np.uint8))

    resized_obs = np.array(resized_obs)
    # Move channel to front
    # moved_axis_observation =  np.moveaxis(resized_obs, 2, 1)
    
    final_observatiom = np.array(resized_obs).astype(np.float32) / 255.0

    return final_observatiom

def chooseActionPong(frames):
    frames = process_pong_frame(frames)
    state_a = np.array([frames], copy=False)
    state_v = torch.tensor(state_a, dtype=torch.float).to("cpu")
    q_vals_v = net(state_v)
    _, act_v = torch.max(q_vals_v, dim=1)
    action = int(act_v.item())
    return action

# our result page view
@csrf_exempt
def get_pong_action(request):
    data = json.loads(request.body)
    board = data['board']
    optimalAction = chooseActionPong(np.array(board))
    return JsonResponse({"action":optimalAction}, safe=False)
    
### CONNECT 4

ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER = 0
AI = 1
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = -1
WINDOW_LENGTH = 4


def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

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

# our result page view
@csrf_exempt
def get_connect_4_action(request):
    data = json.loads(request.body)
    board = np.array(data['board'])
    winner = 0
    if winning_move(board, PLAYER_PIECE):
        winner = PLAYER_PIECE
        return JsonResponse({"action":0,"winner":winner}, safe=False)
    else:
        optimalAction, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
        while(not is_valid_location(board, optimalAction)):
            optimalAction, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
        row = get_next_open_row(board, optimalAction)
        drop_piece(board, row, optimalAction, AI_PIECE)

        if winning_move(board, AI_PIECE):
            winner = AI_PIECE

        return JsonResponse({"action":optimalAction,"winner":winner}, safe=False)
