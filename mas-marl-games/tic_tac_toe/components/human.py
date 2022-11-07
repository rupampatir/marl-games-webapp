import pygame
import components.CONSTANTS as CONSTANTS
import sys

class HumanPlayer:
    def __init__(self, name, player_number=-1):
        self.name = name
        self.player_number = player_number
        self.player_type = 'human'

    def chooseAction_Human(self, positions):
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
          mouseX = event.pos[0] 
          mouseY = event.pos[1] 
          print(mouseX, mouseY)

          clicked_row = int(mouseY // CONSTANTS.BOARD_SQUARE_SIZE)
          clicked_col = int(mouseX // CONSTANTS.BOARD_SQUARE_SIZE)
          action = (clicked_row, clicked_col)
          if action in positions:
              return action
      return -1