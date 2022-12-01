import numpy as np
import random as random
from itertools import product
import pickle
import components.CONSTANTS as CONSTANTS

class Player:
    def __init__(self, name, exp_rate=0.3, player_number=1):
        self.name = name
        self.player_type = 'ai'
        self.player_number = player_number
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9

        ## FOR VS
        self.states_value = {}

        ## FOR Q LEARNING
        permutations = [[-1,0,1] for _ in range(9)]
        states = set(product(*permutations))
        self.actions = []
        if (player_number==1):
          self.player_states =  {state for state in states if (state.count(0)%2 == 1 and state.count(-1)==(state.count(1)))}
        else:
          self.player_states = {state for state in states if (state.count(0)%2 == 0 and state.count(1)==(state.count(-1)+1))} #
        self.Q = self.initialize_Q(self.player_states)
        print(len(self.player_states))
        
    def chooseAction_Random(self, positions):
        idx = np.random.choice(len(positions))
        action = positions[idx]
        return action
      
    def chooseAction_RuleBased(self, availablePositions, board):
      
        row, col = random.choice(availablePositions)

        # check diagonals
        diag_sum1 = sum([board[i, i] for i in range(CONSTANTS.NUM_OF_CELLS)])
        diag_sum2 = sum([board[i, CONSTANTS.NUM_OF_CELLS - i - 1] for i in range(CONSTANTS.NUM_OF_CELLS)])
        
        if diag_sum1 == 2:
            for r, c in [[0, 0], [1, 1], [2, 2]]:
                if [r, c] in availablePositions:
                    row, col = r, c
        elif diag_sum2 == 2:
            for r, c in [[0, 2], [1, 1], [2, 0]]:
                if [r, c] in availablePositions:
                    row, col = r, c

        for r, c in availablePositions:
            if (abs(sum(board[r, :])) == 2 or abs(sum(board[:,c])) == 2):
                row, col = r, c

        return (row, col)

    def getHash(self, board):
      boardHash = str(board.astype(int).reshape(CONSTANTS.NUM_OF_CELLS * CONSTANTS.NUM_OF_CELLS))
      return boardHash

    # append a hash state
    def add_state(self, state, action):
        # print(state, action)
        self.states.append(state)
        self.actions.append(action)

    def reset(self):
        self.states = []
        self.actions = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.Q, fw)
        fw.close()

    def loadPolicy(self, file_Q):
        fr = open(file_Q, 'rb')
        self.Q = pickle.load(fr)
        fr.close()

    def initialize_Q(self, S):
      Q = {}
      for state in S:
          state_hash = self.getHash(np.array(state))
          Q[state_hash]= {}
          for i,x  in enumerate(state):
              r = i//3
              c = i%3
              if x == 0:
                  Q[state_hash][(r,c)] = np.random.rand()
      return Q
    
    def updateQTable(self, reward, alpha = 0.25, gamma = 0.8):
        state, action, newState, newAction = self.states[-1], self.actions[-1], self.states[-2],self.actions[-2],
        if (newState==None or newAction==None):
          self.Q[state][action]+= alpha*(reward - self.Q[state][action])
        else:
          self.Q[state][action]+= alpha*(reward + gamma*self.Q[newState][newAction] - self.Q[state][action])
        
    def chooseAction_QTable(self, positions, current_board):
        # print("Q")
        current_state = self.getHash(current_board)
        best_action = []
        best_action_value = -np.Inf
        for action in positions:
            Q_s_a = self.Q[current_state][action]
            if Q_s_a == best_action_value:
                best_action.append(action)
            elif Q_s_a > best_action_value:
                best_action = [action]
                best_action_value = Q_s_a
        best_action = random.choice(best_action)

        n_actions =len(positions)
        p = np.full(n_actions,self.exp_rate/n_actions)
        p[positions.index(best_action)]+= 1 - self.exp_rate
      
        return positions[np.random.choice(len(positions),p=p)]
    