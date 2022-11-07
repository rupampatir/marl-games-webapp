import random as random
import pygame
import pygame_menu
import sys
from components.human import HumanPlayer
from components.player import Player
from components.board import Board
import components.CONSTANTS as CONSTANTS

class Menu():

  def __init__(self):
    screen = pygame.display.set_mode((CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS, CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS))
    self.game_mode = "Random"
    
    def set_game_mode(value, key):
        # Do the job here !
        self.game_mode = value[0][1]

    def onquit():
      pygame_menu.events.EXIT
      pygame.display.quit()
      pygame.quit()
      sys.exit()
    
    menu = pygame_menu.Menu('Tic Tac Toe', CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS, CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS, theme=pygame_menu.themes.THEME_BLUE)
    # menu.add.text_input('Name :', default='John Doe')
    menu.add.selector('Mode :', [('Random Choice', 'Random'), ('Rule Based', 'RuleBased'), ('Q Learning','Q')], onchange=set_game_mode)
    menu.add.button('Play', self.start_the_game)
    menu.add.button('Quit', onquit)
    menu.mainloop(screen)

  def start_the_game(self):
      human_starts = random.random()>0.5
      player_1, player_2 = None, None
      if (human_starts):
        # print("human")
        player_1 = HumanPlayer("Human", player_number=1)
        player_2 = Player("Computer", exp_rate=0, player_number=-1)
        player_2.loadPolicy("policy_Q_2_both")
      else:
        # print("ai")
        player_1 = Player("Computer", exp_rate=0, player_number=1)
        player_1.loadPolicy("policy_Q_1_both")
        player_2 = HumanPlayer("Human", player_number=-1)

      st = Board(player_1, player_2, 1, self.game_mode)
      st.play_human_vs_AI()