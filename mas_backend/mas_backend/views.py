from django.shortcuts import render
import os
import numpy as np
import random as random
import pickle
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


policy_1_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'policy_Q_1_both')
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