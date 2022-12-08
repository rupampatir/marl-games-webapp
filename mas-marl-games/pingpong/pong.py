import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
from random import randint, choice
import matplotlib.pyplot as plt

WIDTH = 50
HEIGHT = 26
PADDLE_HEIGHT = 3

class Ball(object):
    def __init__(self):
        self.x_ball, self.y_ball = WIDTH/2, randint(0, HEIGHT-1)
        self.x_direction = choice([-1,1])
        self.y_direction = choice([-1,1])
    
    def reset(self):
        self.x_ball, self.y_ball = WIDTH/2, randint(0, HEIGHT-1)
        self.x_direction = choice([-1,1])
        self.y_direction = choice([-1,1])

class Player(object):

    def __init__(self, number=1):
        if number == 0:            
            self.x = 0
            self.y = HEIGHT/2
        if number == 1:
            self.x = WIDTH-1
            self.y = HEIGHT/2
        self.number = number
        self.stay = 0
        self.up = 1
        self.down = 2
        self.won = False
        
    def init_player(self):
        if self.number == 0: 
            self.x = 0
            self.y = HEIGHT/2
        else:
            self.x = WIDTH-1
            self.y = HEIGHT/2
        self.won = False

    def do_move(self, move):
        if self.y>0 and move == self.up:
            self.y -= 1
        elif self.y+PADDLE_HEIGHT<HEIGHT and move == self.down:
            self.y += 1

import numpy as np
import matplotlib.pyplot as plt

class Game:

    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.player = []
        self.ball = []
        self.game_over = False

    def reset(self):
        for p in self.player:
            p.init_player()
        for ball in self.ball:
            ball.reset()
        self.game_over = False

    
    def perform_steps(self, actions, is_train = True):
        for i in range(len(actions)):
            self.player[i].do_move(actions[i])
        
        new_ball_x = self.ball[0].x_ball+self.ball[0].x_direction
        new_ball_y = self.ball[0].y_ball+self.ball[0].y_direction
        if (new_ball_x<=0):
            if (is_train or (new_ball_y>=self.player[1].y and new_ball_y<=(self.player[1].y+PADDLE_HEIGHT))):
                self.ball[0].x_direction*=-1
        
        if (new_ball_x>=WIDTH-1 and (new_ball_y>=self.player[1].y and new_ball_y<=(self.player[1].y+PADDLE_HEIGHT))):
            self.ball[0].x_direction*=-1

        if (new_ball_y<=0 or new_ball_y>=HEIGHT-1):
            self.ball[0].y_direction*=-1
        

        self.ball[0].x_ball+=self.ball[0].x_direction
        self.ball[0].y_ball+=self.ball[0].y_direction
        if (self.ball[0].x_ball<0):
            self.game_over = True
            self.player[1].won = True
        elif (self.ball[0].x_ball>=WIDTH):
            self.game_over = True
            self.player[0].won = True
        return self.game_over

    def get_game_matrix(self):
        game_matrix = np.zeros(shape=(HEIGHT, WIDTH))
        for p in game.player:
            for j in range(PADDLE_HEIGHT):
                game_matrix[int(p.y)+j, int(p.x)] = 255
        for ball in game.ball:
            game_matrix[int(ball.y_ball), int(ball.x_ball)] = 255

        # plt.imshow(game_matrix)
        # plt.colorbar()
        # plt.show(block=False)
        # plt.pause(1.0/24)
        # plt.close()
        return game_matrix
    
class DQNAgent(object):

    def __init__(self, weights=False, dim_state=6, gamma=0.9, learning_rate=0.0005):
        self.dim_state = dim_state
        self.reward = 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.network()
        if weights:
            self.model = self.network(weights)
        self.memory = []
        self.name = "dqn"

    def get_state(self, game, player, ball):

        state = [
            ball.y_ball > player.y+PADDLE_HEIGHT,
            ball.y_ball < player.y,
            ball.x_direction < 0,
            ball.x_direction > 0,
            ball.y_direction < 0,
            ball.y_direction > 0
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state).reshape(1, self.dim_state)

    def set_reward(self, player):
        self.reward = 0
        if player.won:
            self.reward = 10
        else:
            self.reward = -10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(120, activation='relu', input_dim=self.dim_state))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(3, activation='softmax'))  # [right, left, up, down]
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
    epsilon = 1
    plot_data = []
    # decay factor = (Final epsilon value / Initial epsilon value)^(1/number of episodes)
    decay_factor = (0.01/1.0)**(1/max_episodes)
    while episodes <= max_episodes:
        game.reset()
        steps = 0
        total_reward = 0
        while not game.game_over:            
            state_old = agent.get_state(game, game.player[1], game.ball[0])
            if random.random() < epsilon:
                move = randint(0, 2)
            else:
                prediction = agent.model.predict(state_old)
                move = np.argmax(prediction[0])

            game_over = game.perform_steps([0,move])
            state_new = agent.get_state(game, game.player[1], game.ball[0])
            reward = 0 if not game_over else agent.set_reward(game.player[1])
            agent.train_online_net(state_old, move, reward, state_new, game_over)
            agent.remember(state_old, move, reward, state_new, game_over)
            steps += 1
            if (reward!=0):
                plot_data.append([episodes, reward])
            total_reward+=reward
            if (not game.game_over):
                game.get_game_matrix()
        print('Episode:', episodes, 'Reward:', reward)
        epsilon*=decay_factor
        # agent.replay_mem(agent.memory)
        episodes += 1
        if episodes % 100 == 0:
            agent.model.save_weights("PING PONG" + str(episodes) + '.hdf5')
            print('Episode:', episodes, 'Win Rate:', np.average(np.array(plot_data)[:,-100:]))

    return plot_data

game = Game()
game.player.append(Player(0))
game.player.append(Player(1))
game.ball.append(Ball())
train_dqn(game,1000)