import pygame
from components.player import Player
from components.board import Board
import matplotlib.pyplot as plt
import numpy as np

pygame.init()
figure_count = 0

def get_plot(plot_data, algo):
    global figure_count

    plt.figure(figure_count)
    plot_data = np.array(plot_data)
    x = plot_data[:,0]
    y = plot_data[:,1]
    window = 100 #int(len(y)/50)
    average_y = []
    for ind in range(len(y) - window + 1):
         average_y.append(np.mean(y[ind:ind+window]))
    
    for ind in range(window - 1):
        average_y.insert(0, np.nan)
    
    plot_data = np.array(plot_data)
    plt.plot(x,average_y)
    plt.ylabel('Winner')
    plt.xlabel('Episode')
    plt.title(algo + ": Episode vs Winner [Running Average]")
    # plt.show()
    plt.savefig(algo + "(" + str(len(x)) + " Episodes) : Episode vs Winner [Running Average]")
    figure_count+=1


    

# training
### VALUE STATE (FIRST MOVE) ###
# player_1 = Player("VS_1", player_number=1)
# player_2 = Player("discard", player_number=-1)

# st = Board(player_1, player_2, 1, "VS")
# print("training...")
# st.train_RL(10000, 'first')

# player_1.savePolicy()

### VALUE STATE (OPPONENT) ###

# player_1 = Player("random", player_number=1)
# player_2 = Player("VS_2", player_number=-1)

# st = Board(player_1, player_2, 1, "VS")
# print("training...")
# st.train_RL(20000, 'opponent')

# player_2.savePolicy()


### Q LEARNING (FIRST MOVE) ###

# win_rate = 0
# win_rates = []

# for episodes in [1000,2000,5000,10000,15000,20000]:
#     player_1 = Player("Q_1", player_number=1)
#     player_2 = Player("random", player_number=-1)

#     st = Board(player_1, player_2, 1, "Q")
#     print("training...")
#     plot_data = st.train_RL(episodes, 'first')

#     get_plot(plot_data, "DQN Player 1")

#     y = list(np.array(plot_data)[:,1])

#     print("Player 1: " + str(len(y)) + " Episodes")
#     print("Win rate", 1.0*y.count(1)/len(y))
#     print("Draw rate", 1.0*y.count(0)/len(y))
#     print("Lose rate", 1.0*y.count(-1)/len(y))
#     win_rates.append([episodes, 1.0*y.count(1)/len(y)])
#     if (1.0*y.count(1)/len(y)>win_rate):
#         player_1.savePolicy()

# win_rates = np.array(win_rates)
# print
# x = win_rates[:,0]
# y = win_rates[:,1]
# plt.figure(figure_count)
# plt.plot(x,y)
# plt.ylabel('Rate')
# plt.xlabel('Number of episodes during Training')
# plt.xticks(x)
# plt.title("Player 1: Win Rate vs Training Episodes")
# plt.savefig("Player 1: Win Rate vs Training Episodes")
# figure_count+=1

# print(x)
# print(y)

# ### Q LEARNING (OPPONENT) ###


# win_rate = 0
# win_rates = []

# for episodes in [5000,10000,15000,20000, 25000, 30000]:
    
#     player_1 = Player("random", player_number=1)
#     player_2 = Player("Q_2", player_number=-1)

#     st = Board(player_1, player_2, 1, "Q")
#     print("training...")
#     plot_data = st.train_RL(episodes, 'opponent')

#     get_plot(plot_data, "DQN Player 2")

#     y = list(np.array(plot_data)[:,1])

#     print("Player 2: " + str(len(y)) + " Episodes")
#     print("Win rate", 1.0*y.count(1)/len(y))
#     print("Draw rate", 1.0*y.count(0)/len(y))
#     print("Lose rate", 1.0*y.count(-1)/len(y))
#     win_rates.append([episodes, 1.0*y.count(1)/len(y)])
#     if (1.0*y.count(1)/len(y)>win_rate):
#         player_2.savePolicy()

# win_rates = np.array(win_rates)
# x = win_rates[:,0]
# y = win_rates[:,1]
# plt.figure(figure_count)
# plt.plot(x,y)
# plt.ylabel('Rate')
# plt.xlabel('Number of episodes during Training')
# plt.xticks(x)
# plt.title("Player 2: Win Rate vs Training Episodes")
# plt.savefig("Player 2: Win Rate vs Training Episodes")
# figure_count+=1

# print(x)
# print(y)
# [ 5000. 10000. 15000. 20000. 25000. 30000.]
# [0.3876     0.3191     0.3106     0.2632     0.30052    0.26936667]

## Train both

win_rate_1 = 0
win_rates_1 = []
win_rate_2 = 0
win_rates_2 = []

for episodes in [1000,2000,5000,10000,15000,20000]:
    player_1 = Player("Q_1_both", player_number=1)
    player_2 = Player("Q_2_both", player_number=-1)
    player_2.loadPolicy("policy_Q_1")
    player_2.loadPolicy("policy_Q_2")

    st = Board(player_1, player_2, 1, "Q")
    print("training...")
    plot_data = st.train_RL(episodes, 'both')

    get_plot(plot_data, "DQN Train Both")

    y = list(np.array(plot_data)[:,1])

    win_rates_1.append([episodes, 1.0*y.count(1)/len(y)])

    if (1.0*y.count(1)/len(y)>win_rate_1):
        player_1.savePolicy()
        win_rate_1 = 1.0*y.count(1)/len(y)
    if (1.0*y.count(1)/len(y)>win_rate_2):
        player_2.savePolicy()
        win_rate_2 = 1.0*y.count(1)/len(y)

win_rates_1 = np.array(win_rates_1)
print
x = win_rates_1[:,0]
y = win_rates_1[:,1]
plt.figure(figure_count)
plt.plot(x,y)
plt.ylabel('Rate')
plt.xlabel('Number of episodes during Training')
plt.xticks(x)
plt.title("Player 1: Win Rate vs Training Episodes")
plt.savefig("Player 1: Win Rate vs Training Episodes")
figure_count+=1

print(x)
print(y)