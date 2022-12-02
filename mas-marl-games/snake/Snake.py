import pygame
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Food(object):

    def __init__(self, game):
        self.x_food, self.y_food = 0, 0
        self.food_coord(game)
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game):
        self.x_food, self.y_food = game.find_free_space()

    def display_food(self, game):
        game.gameDisplay.blit(self.image, (self.x_food, self.y_food))
        pygame.display.update()


class Player(object):

    def __init__(self, game, color="green"):
        self.color = color
        if self.color == "red":
            self.image = pygame.image.load('img/redsnake.png')
            x = 0.3 * game.game_width
            y = 0.3 * game.game_height
        if self.color == "blue":
            self.image = pygame.image.load('img/bluesnake.png')
            x = 0.3 * game.game_width
            y = 0.7 * game.game_height
        # if self.color == "red":
        #     self.image = pygame.image.load('img/snakeBody3.png')
        #     x = 0.7 * game.game_width
        #     y = 0.3 * game.game_height
        # if self.color == "purple":
        #     self.image = pygame.image.load('img/snakeBody4.png')
        #     x = 0.7 * game.game_width
        #     y = 0.7 * game.game_height
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

    def display_player(self, game):
        self.position[-1][0] = self.x
        self.position[-1][1] = self.y

        if not self.crash:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))
            pygame.display.update()
        else:
            game.end = True

    
    def select_move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]
        state = self.agent.get_state(game, self, food)
        prediction = self.agent.model.predict(state)
        move = np.argmax(prediction[0])
        return move

    def move_as_array(self, move):
        if move == self.right:
            return [1, 0, 0, 0]
        elif move == self.left:
            return [0, 1, 0, 0]
        elif move == self.up:
            return [0, 0, 1, 0]
        elif move == self.down:
            return [0, 0, 0, 1]

    def set_agent(self, agent):
        self.agent = agent

    def distance_closest_food(self, game, x, y):
        distance = []
        for food in game.food:
            distance.append(abs(x - food.x_food/self.step_size) + abs(y - food.y_food/self.step_size))
        return distance[np.argmin(distance)]


class Game:

    def __init__(self, width=20, height=20, game_speed=30, display_option=True):
        self.game_width = width * 20 + 40
        self.game_height = height * 20 + 40
        self.width = width
        self.height = height
        if display_option:
            self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height + 100))
            self.bg = pygame.image.load("img/background.png")
            pygame.display.set_caption('Snake')
        self.player = []
        self.food = []
        self.display_option = display_option
        self.game_speed = game_speed

    # return the coordinates of a location without snakes' parts or walls
    def find_free_space(self):
        x_rand = randint(20, self.game_width - 40)
        x = x_rand - x_rand % 20
        y_rand = randint(20, self.game_height - 40)
        y = y_rand - y_rand % 20
        for player in self.player:
            if [x, y] not in player.position:
                return x, y
            else:
                return self.find_free_space()

    def display(self):
        if self.display_option:
            self.gameDisplay.fill((255, 255, 255))
            for player in self.player:
                player.display_player(self)
            for food in self.food:
                food.display_food(self)


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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_mem(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]).reshape((1, self.dim_state)))[0])
            target_f = self.model.predict(np.array([state]).reshape((1, self.dim_state)))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]).reshape((1, self.dim_state)), target_f, epochs=1, verbose=0)

    def train_online_net(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.dim_state)))[0])
        target_f = self.model.predict(state.reshape((1, self.dim_state)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.dim_state)), target_f, epochs=1, verbose=0)


# train model
def train_dqn(game,max_episodes=300):
    agent = DQNAgent()
    episodes = 0   
    max_score = 0
    epsilon = 1
    plot_data = []
    # decay factor = (Final epsilon value / Initial epsilon value)^(1/number of episodes)
    decay_factor = (0.01/1.0)**(1/episodes)
    while episodes <= max_episodes:
        game.player[0].init_player(game)
        steps = 0
        while not game.player[0].crash:            
            state_old = agent.get_state(game, game.player[0], game.food[0])
            if random.random < epsilon:
                move = randint(0, 3)
            else:
                prediction = agent.model.predict(state_old)
                move = np.argmax(prediction[0])
            action = game.player[0].move_as_array(move)

            game.player[0].do_move(move, game)
            state_new = agent.get_state(game, game.player[0], game.food[0])
            reward = agent.set_reward(game.player[0])
            agent.train_online_net(state_old, action, reward, state_new, game.player[0].crash)
            agent.remember(state_old, action, reward, state_new, game.player[0].crash)
            steps += 1
            if game.player[0].eaten:
                steps = 0
            if steps >= 1000:
                game.player[0].crash = True
            if game.player[0].score > max_score:
                max_score = game.player[0].score

        epsilon*=decay_factor
        agent.replay_mem(agent.memory)
        episodes += 1
        print('Episode:', episodes, 'Max Score:', max_score)
        plot_data.append(game.player[0].score)
        if episodes % 100 == 0:
            agent.model.save_weights("SNAKE" + str(episodes) + '.hdf5')
    return plot_data

game = Game(20, 20)
game.player.append(Player(game, "green"))
game.food.append(Food(game))
game.display_option = False
game.game_speed = 30
train_dqn(game,300)

# if __name__ == "__main__":
#     pygame.init()
#     pygame.font.init()
#     game = Game(20, 20)

#     snake_blu = Player(game, "blue")
#     rl_agent = DQNAgent('weights/dqn_algorithm/weights_snake_300.hdf5')
#     snake_blu.set_agent(rl_agent)
#     game.player.append(snake_blu)
    
    
#     snake_red = Player(game, "red")
#     rl_agent2 = DQNAgent('weights/dqn_algorithm/weights_snake_300.hdf5')
#     snake_red.set_agent(rl_agent2)
#     game.player.append(snake_red)

#     game.food.append(Food(game))
    
#     game.game_speed = 0
#     game.display_option = True
#     frames = []
#     for i in range(len(game.player)):
#         move = game.player[i].select_move(game)
#         game.player[i].do_move(move, game)
#         if game.player[i].crash:
#             game.player[i].init_player(game)

#     game.display()
#     pygame.time.wait(game.game_speed)
