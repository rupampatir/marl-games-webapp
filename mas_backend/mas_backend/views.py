from django.shortcuts import render
import os
import numpy as np
import random as random
import pickle
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import math
from tensorflow.keras.optimizers import Adam # - Works
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

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


# import numpy as np
# import pickle
# import time

# # weights initialization
# previously_processed_frame = None
# pong_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn_weights.pkl')
# pong_weights = pickle.load(open(pong_weights, "rb"))

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]
# from PIL import Image

# def process_frame(frame, previously_processed_frame):
#     # crop
#     processed_frame = frame[35:195]
#     # reduce image size
#     processed_frame = processed_frame[::2, ::2, :]
#     # remove color
#     processed_frame = processed_frame[:, :, 0]
#     # remove background
#     processed_frame[processed_frame == 144] = 0
#     processed_frame[processed_frame == 109] = 0
#     # set everything that's not black to white
#     processed_frame[processed_frame != 0] = 1
#     # print(processed_frame)
#     # im = Image.fromarray(np.array(processed_frame.astype(np.uint8)))
#     # im.save("image.jpeg")
#     # flatten frame
#     processed_frame = processed_frame.astype(np.float).ravel()
    
#     # flatten frame
#     # subtract the previous frame from the current one so we are only processing
#     # on changes in the game
#     if previously_processed_frame is not None:
#       delta_frame = processed_frame - previously_processed_frame
#     else:
#       # the below flattens a frame of 0s
#       delta_frame = np.zeros(processed_frame.size)
#     # store the previous frame so we can subtract from it next time
#     previously_processed_frame = processed_frame
#     return delta_frame, previously_processed_frame

# def forward(frame):
#     hidden_layer = np.dot(pong_weights['W1'], frame)
#     hidden_layer[hidden_layer < 0] = 0 # ReLU nonlinearity
#     yhat = np.dot(pong_weights['W2'], hidden_layer)
#     up_probability = sigmoid(yhat)
#     return hidden_layer, up_probability # return probability of taking action 2, and hidden state

# def choose_action(probability):
#     """ this is the stochastic part of RL and what makes this policy
#     neural network different from a supervised learning problem """
#     random_value = np.random.uniform()
#     if random_value < probability:
#       # signifies up in openai gym
#       return 2
#     else:
#        # signifies down in openai gym
#       return 3

@csrf_exempt
def get_pong_action(request):
    pass
#     global previously_processed_frame
#     data = json.loads(request.body)
#     current_frame = np.array(data['frame'])
#     delta_frame, previously_processed_frame = process_frame(current_frame, previously_processed_frame)
#     # forward the policy network and sample an action from the returned probability
#     hidden_layer, up_probability = forward(delta_frame)
#     optimalAction = choose_action(up_probability)
#     return JsonResponse({"action":optimalAction}, safe=False)
    






### Snake

import numpy as np
# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout
import random
# from random import randint
# import matplotlib.pyplot as plt
# import seaborn as sns
import os

class Food(object):

    def __init__(self, game):
        self.x_food, self.y_food = 0, 0
        self.food_coord(game)

    def food_coord(self, game):
        self.x_food, self.y_food = game.find_free_space()

class Player(object):

    def __init__(self, game, color="green"):
        self.color = color
        if self.color == "red":
            x = 0.3 * game.game_width
            y = 0.3 * game.game_height
            self.player_number = 1
        if self.color == "blue":
            x = 0.3 * game.game_width
            y = 0.7 * game.game_height
            self.player_number = 2
        if self.color == "purple":
            x = 0.7 * game.game_width
            y = 0.3 * game.game_height
            self.player_number = 3
        if self.color == "green":
            x = 0.7 * game.game_width
            y = 0.7 * game.game_height
            self.player_number = 4
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []  # coordinates of all the parts of the snake
        self.position.append([self.x, self.y])  # append the head
        self.food = 1  # length
        self.eaten = False
        self.right = 0
        self.left = 1
        self.up = 2
        self.down = 3
        self.direction = self.right
        self.step_size = 20  # pixels per step
        self.crash = False
        self.score = 0
        self.record = 0
        self.deaths = 0
        self.total_score = 0  # accumulated score
        self.agent = None
        
    def init_player(self, game):
        self.x, self.y = game.find_free_space()
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.direction = self.right
        self.step_size = 20
        self.crash = False
        self.score = 0

    def update_position(self):
        if self.position[-1][0] != self.x or self.position[-1][1] != self.y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = self.x
            self.position[-1][1] = self.y

    def eat(self, game):
        for food in game.food:
            if self.x == food.x_food and self.y == food.y_food:
                food.food_coord(game)
                self.eaten = True
                self.score += 1
                self.total_score += 1

    def crushed(self, game, x=-1, y=-1):
        if x == -1 and y == -1:  # coordinates of the head
            x = self.x
            y = self.y
        if x < 20 or x > game.game_width - 40 \
                or y < 20 or y > game.game_height - 40:
            return True
        for player in game.player:
            if [x, y] in player.position:
                return True

    def do_move(self, move, game):
        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        move_array = [0, 0]

        if move == self.right:
            move_array = [self.step_size, 0]
        elif move == self.left:
            move_array = [-self.step_size, 0]
        elif move == self.up:
            move_array = [0, -self.step_size]
        elif move == self.down:
            move_array = [0, self.step_size]

        if move == self.right and self.direction != self.left:
            move_array = [self.step_size, 0]
            self.direction = self.right
        elif move == self.left and self.direction != self.right:
            move_array = [-self.step_size, 0]
            self.direction = self.left
        elif move == self.up and self.direction != self.down:
            move_array = [0, -self.step_size]
            self.direction = self.up
        elif move == self.down and self.direction != self.up:
            move_array = [0, self.step_size]
            self.direction = self.down
        self.x += move_array[0]
        self.y += move_array[1]

        if self.crushed(game):
            self.crash = True
            self.deaths += 1
            if self.score > self.record:
                self.record = self.score

        self.eat(game)
        self.update_position()
   
    def select_move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]
        state = self.agent.get_state(game, self, food)
        prediction = self.agent.model.predict(state)
        move = np.argmax(prediction[0])
        return move
	
    def set_agent(self, agent):
        self.agent = agent

class Game:

    def __init__(self, width=20, height=20, game_speed=30):
        self.game_width = width * 20 + 40
        self.game_height = height * 20 + 40
        self.width = width
        self.height = height
        self.player = []
        self.food = []
        self.game_speed = game_speed

    # return the coordinates of a location without snakes' parts or walls
    def find_free_space(self):
        x_rand = random.randint(20, self.game_width - 40)
        x = x_rand - x_rand % 20
        y_rand = random.randint(20, self.game_height - 40)
        y = y_rand - y_rand % 20
        for player in self.player:
            if [x, y] not in player.position:
                return x, y
            else:
                return self.find_free_space()
	
    def get_board_state(self):
        board = [[0 for i in range(20)] for j in range(20)]
        for player in self.player:
            for x, y  in player.position:
                x = int(x/20)
                y = int(y/20)
                if (x<len(board) and y<len(board[0])):
                    board[x][y] = player.player_number
        for food in self.food:
            x = int(food.x_food/20)
            y = int(food.y_food/20)
            if (x<len(board) and y<len(board[0])):
              board[x][y] = -1
        
        return board

    def get_player_scores(self):
        scores = []
        for player in self.player:
            scores.append(player.score)
        return scores

    def closest_food(self, player):
        distance = []
        for food in self.food:
            distance.append(abs(player.x - food.x_food) + abs(player.y - food.y_food))
        return np.argmin(distance)


class DQNAgent(object):

    def __init__(self, weights=False, dim_state=12, gamma=0.9, learning_rate=0.0005):
        self.dim_state = dim_state
        self.reward = 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.network()
        if weights:
            self.model = self.network(weights)
        self.memory = []
        self.name = "dqn"

    def get_state(self, game, player, food):

        game_matrix = np.zeros(shape=(game.width+2, game.height+2))
        for p in game.player:
            for i, coord in enumerate(p.position):
                game_matrix[int(coord[1]/game.width), int(coord[0]/game.height)] = 1
        for food in game.food:
            game_matrix[int(food.y_food/game.width), int(food.x_food/game.height)] = 2
        for i in range(game.width+2):
            for j in range(game.height+2):
                if i == 0 or j == 0 or i == game.width+1 or j == game.height+1:
                    game_matrix[i, j] = 1
        head = player.position[-1]
        player_x, player_y = int(head[0]/game.width), int(head[1]/game.height)

        #print(game_matrix)
        food = game.food[game.closest_food(player)]
        state = [
            player_x + 1 < game.width+2 and game_matrix[player_y, player_x+1] == 1,  # danger right
            player_x + -1 >= 0 and game_matrix[player_y, player_x-1] == 1,  # danger left
            player_y + -1 >= 0 and game_matrix[player_y-1, player_x] == 1,  # danger up
            player_y + 1 < game.height+2 and game_matrix[player_y+1, player_x] == 1,  # danger down
            player.direction == player.right,
            player.direction == player.left,
            player.direction == player.up,
            player.direction == player.down,
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state).reshape(1, self.dim_state)

    def set_reward(self, player):
        self.reward = 0
        if player.crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(120, activation='relu', input_dim=self.dim_state))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(4, activation='softmax'))  # [right, left, up, down]
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model



snake_game = None
@csrf_exempt
def reset_snake_game(request):
    global snake_game
    snake_game = Game(20, 20)
    policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SNAKE_300.hdf5')

    snake_red = Player(snake_game, "red")
    snake_game.player.append(snake_red)

    data = json.loads(request.body)
    num_food = data['num_food']
    num_opponents = data['num_opponents']
    ai_colors = ["green", "purple", "blue"]
    for _ in range(num_opponents):
        snake_ai = Player(snake_game, ai_colors.pop())
        snake_ai.set_agent(DQNAgent(policy_path))
        snake_game.player.append(snake_ai)
    for _ in range(num_food):
        snake_game.food.append(Food(snake_game))
    print(snake_game.player)
    snake_game.game_speed = 0

    return JsonResponse({"board":snake_game.get_board_state(), "scores": snake_game.get_player_scores() }, safe=False) 


@csrf_exempt
def get_snake_action(request):
    global snake_game

    data = json.loads(request.body)
    player_move = np.array(data['action'])
    for player in snake_game.player:
        if (player.player_number == 1):
            player.do_move(player_move, snake_game)
        else:
            ai_move = player.select_move(snake_game)
            player.do_move(ai_move, snake_game)
        if player.crash:
            player.init_player(snake_game)            

    return JsonResponse({"board":snake_game.get_board_state(), "scores": snake_game.get_player_scores() }, safe=False)
    