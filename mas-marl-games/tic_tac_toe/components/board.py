import components.CONSTANTS as CONSTANTS
import pygame
import numpy as np

class Board():
  
  def __init__(self, player_1, player_2, start_player, mode = "Random"):
    self.board_height = CONSTANTS.BOARD_SQUARE_SIZE * CONSTANTS.NUM_OF_CELLS
    self.board_width = CONSTANTS.BOARD_SQUARE_SIZE * CONSTANTS.NUM_OF_CELLS
    self.line_width = 20
    self.padding = 20
    self.is_game_over = False
    self.player_1 = player_1
    self.player_2 = player_2
    self.start_player = start_player
    self.current_player = start_player
    self.board_hash = None
    self.font = pygame.font.SysFont(None, 60)
    # Initalise screen
    self.screen = pygame.display.set_mode((self.board_width, self.board_height))
    pygame.display.set_caption( 'MultiAgent Systems: Tic Tac Toe' )
    self.screen.fill(CONSTANTS.color_bg)

    self.board = np.zeros((CONSTANTS.NUM_OF_CELLS, CONSTANTS.NUM_OF_CELLS))
    self.game_mode = mode

  ### GUI ###

  def draw_board(self):
    pygame.draw.line(self.screen, CONSTANTS.color_line, (0, CONSTANTS.BOARD_SQUARE_SIZE), (self.board_width, CONSTANTS.BOARD_SQUARE_SIZE), self.line_width )
    pygame.draw.line(self.screen, CONSTANTS.color_line, (0, 2 * CONSTANTS.BOARD_SQUARE_SIZE), (self.board_width, 2 * CONSTANTS.BOARD_SQUARE_SIZE), self.line_width )
    pygame.draw.line(self.screen, CONSTANTS.color_line, (CONSTANTS.BOARD_SQUARE_SIZE, 0), (CONSTANTS.BOARD_SQUARE_SIZE, self.board_height), self.line_width )
    pygame.draw.line(self.screen, CONSTANTS.color_line, (2 * CONSTANTS.BOARD_SQUARE_SIZE, 0), (2 * CONSTANTS.BOARD_SQUARE_SIZE, self.board_width), self.line_width )

  def draw_marks(self):
    padding = 20
    circle_radius = CONSTANTS.BOARD_SQUARE_SIZE//2 - 10
    for row in range(len(self.board)):
      for column in range(len(self.board[0])):
        cell = self.board[row][column]
        if (cell == 1):
          pygame.draw.circle(self.screen, CONSTANTS.color_circle, (int(column * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE//2 ), int( row * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE//2 )), circle_radius, self.line_width )
        elif cell == -1:
          pygame.draw.line(self.screen, CONSTANTS.color_cross, (column * CONSTANTS.BOARD_SQUARE_SIZE + self.padding, row * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE - self.padding), (column * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE - padding, row * CONSTANTS.BOARD_SQUARE_SIZE + padding), self.line_width )	
          pygame.draw.line(self.screen, CONSTANTS.color_cross, (column * CONSTANTS.BOARD_SQUARE_SIZE + self.padding, row * CONSTANTS.BOARD_SQUARE_SIZE + self.padding), (column * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE - padding, row * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE - padding), self.line_width )
  
  def draw_win_text(self, text):
    img = self.font.render(text, True, CONSTANTS.color_golden)
    # self.screen.blit(img, (CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS//2-200, CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS//2))
    # # draw text
    # font = pygame.font.Font(None, 25)
    # text = font.render("You win!", True, BLACK)
    img_rect = img.get_rect(center=(CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS/2, CONSTANTS.BOARD_SQUARE_SIZE*CONSTANTS.NUM_OF_CELLS/2))
    self.screen.blit(img, img_rect)
    pygame.display.update()

  

  def draw_vertical_winning_line(self, col, player):
    posX = col * CONSTANTS.BOARD_SQUARE_SIZE + CONSTANTS.BOARD_SQUARE_SIZE//2

    color = CONSTANTS.color_circle
    if player == -1:
      color = CONSTANTS.color_cross

    pygame.draw.line( self.screen, color, (posX, 15), (posX, self.board_height - 15), self.line_width )
    pygame.display.update()

  def draw_horizontal_winning_line(self,row, player):
    posY = row * CONSTANTS.BOARD_SQUARE_SIZE  + CONSTANTS.BOARD_SQUARE_SIZE //2

    color = CONSTANTS.color_circle
    if player == -1:
      color = CONSTANTS.color_cross

    pygame.draw.line( self.screen, color, (15, posY), (self.board_width - 15, posY), self.line_width )
    pygame.display.update()

  def draw_asc_diagonal(self,player):
    color = CONSTANTS.color_circle
    if player == -1:
      color = CONSTANTS.color_cross

    pygame.draw.line(self.screen, color, (15, self.board_height - 15), (self.board_width - 15, 15), self.line_width )
    pygame.display.update()

  def draw_desc_diagonal(self,player):
    color = CONSTANTS.color_circle
    if player == -1:
      color = CONSTANTS.color_cross

    pygame.draw.line( self.screen, color, (15, 15), (self.board_width - 15, self.board_height - 15), self.line_width )
    pygame.display.update()



  ### UTILITIES ###


  def get_hash(self):
    self.board_hash = str(self.board.astype(int).reshape(CONSTANTS.NUM_OF_CELLS * CONSTANTS.NUM_OF_CELLS))
    return self.board_hash

  def get_available_squares(self):
    available_squares = []
    for row in range(CONSTANTS.NUM_OF_CELLS):
      for column in range(CONSTANTS.NUM_OF_CELLS):
        if self.board[row][column] == 0:
          available_squares.append((row,column))
    return available_squares
  
  def check_for_win(self):
        
        # check across
        for i in range(CONSTANTS.NUM_OF_CELLS):
            if sum(self.board[i, :]) == CONSTANTS.NUM_OF_CELLS:
                self.draw_horizontal_winning_line(i, 1)
                return 1
            if sum(self.board[i, :]) == -CONSTANTS.NUM_OF_CELLS:
                self.draw_horizontal_winning_line(i, -1)
                return -1
        
        # check down
        for i in range(CONSTANTS.NUM_OF_CELLS):
            if sum(self.board[:, i]) == CONSTANTS.NUM_OF_CELLS:
                self.draw_vertical_winning_line(i, 1)
                return 1
            if sum(self.board[:, i]) == -CONSTANTS.NUM_OF_CELLS:
                self.draw_vertical_winning_line(i, -1)
                return -1

        # check diagonals
        diag_sum1 = sum([self.board[i, i] for i in range(CONSTANTS.NUM_OF_CELLS)])
        diag_sum2 = sum([self.board[i, CONSTANTS.NUM_OF_CELLS - i - 1] for i in range(CONSTANTS.NUM_OF_CELLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == CONSTANTS.NUM_OF_CELLS:
            if diag_sum1 == CONSTANTS.NUM_OF_CELLS:
              self.draw_desc_diagonal(1)
              return 1
            elif diag_sum2 == CONSTANTS.NUM_OF_CELLS:
              self.draw_asc_diagonal(1)
              return 1
            elif diag_sum1 == -CONSTANTS.NUM_OF_CELLS:
              self.draw_desc_diagonal(-1)
              return -1
            elif diag_sum2 == -CONSTANTS.NUM_OF_CELLS:
              self.draw_asc_diagonal(-1)
              return -1

        if self.is_board_full():
            return 0
        
        return None
  
  def update_state(self, position):
    self.board[position] = self.current_player
    # switch to another player
    self.current_player = -1 if self.current_player == 1 else 1

  def mark_square(self, row, column, player):
    self.board[row][column] = player
    self.current_player = -1 if self.current_player == 1 else 1
  
  def is_board_full(self):
    for row in range(CONSTANTS.NUM_OF_CELLS):
      for column in range(CONSTANTS.NUM_OF_CELLS):
        if self.board[row][column] == 0:
          return False
    return True
  
  def render(self):
    self.screen.fill(CONSTANTS.color_bg)
    self.draw_board()
    self.draw_marks()
    pygame.display.update()

  def reset(self):
    self.board = np.zeros((CONSTANTS.NUM_OF_CELLS, CONSTANTS.NUM_OF_CELLS))
    self.board_hash = None
    self.is_game_over = False
    self.current_player = self.start_player
    self.render()
  

  ### LEARNING ###

  def giveReward(self):
      
    result = self.check_for_win()
    if result == 1:
        self.player_1.updateQTable(1)
        self.player_2.updateQTable(-1)
    elif result == -1:
        self.player_1.updateQTable(-1)
        self.player_2.updateQTable(1)
    else:
        self.player_1.updateQTable(0.1)
        self.player_2.updateQTable(0.5)
        
  def train_RL(self, rounds=100, train_type='opponent'):

    plot_data = []

    for i in range(rounds):
        if i % 1000 == 0:
            print("Rounds {}".format(i))
        if (self.game_mode=="Q"):
          self.player_1.add_state(None, None)
          self.player_2.add_state(None, None)
          
        while not self.is_game_over:
            # Player 1
            positions = self.get_available_squares()
            player_1_action = 0
            if (train_type=='opponent'):
              player_1_action = self.player_1.chooseAction_Random(positions)
            else:
              player_1_action = self.player_1.chooseAction_QTable(positions, self.board)
            
            # take action and upate board state
            board_hash = self.get_hash()
            self.player_1.add_state(board_hash, player_1_action)
            self.update_state(player_1_action)
            # check board status if it is end

            win = self.check_for_win()
            if win is not None:
                # self.showBoard()
                # ended with player_1 either win or draw
                plot_data.append([i,win])
                self.giveReward()
                self.player_1.reset()
                self.player_2.reset()
                self.reset()
                break
            else:
                # Player 2
                positions = self.get_available_squares()
                player_2_action = 0
                if (train_type=='first'):
                  player_2_action = self.player_2.chooseAction_Random(positions)
                else:
                  player_2_action = self.player_2.chooseAction_QTable(positions, self.board)
                board_hash = self.get_hash()
                self.player_2.add_state(board_hash, player_2_action)
                self.update_state(player_2_action)
                win = self.check_for_win()
                if win is not None:
                  plot_data.append([i,win])
                  self.giveReward()
                  self.player_1.reset()
                  self.player_2.reset()
                  self.reset()
                  break
    return plot_data

  # play with human
  def play_human_vs_AI(self):
      
      while not self.is_game_over:
          self.render()
          if (self.current_player==1):
            
            # Player 1
            positions = self.get_available_squares()
            player_1_action = 0
            if (self.player_1.player_type == 'human'):
                player_1_action = self.player_1.chooseAction_Human(positions)
            else:
              if (self.game_mode == "Random"):
                player_1_action = self.player_1.chooseAction_Random(positions)
                print("random")
              elif(self.game_mode == "RuleBased"):
                print("RB")
                player_1_action = self.player_1.chooseAction_RuleBased(positions, self.board)
              else:
                print("RL")
                player_1_action = self.player_1.chooseAction_QTable(positions, self.board)
            
            if (player_1_action!=-1):
              self.update_state(player_1_action)
              self.render()
              win = self.check_for_win()
              if win is not None:
                  pygame.time.wait(500)
                  if win == 1:
                      self.draw_win_text(self.player_1.name + " wins!")
                  else:
                      self.draw_win_text("Tie!")
                  # is_menu = True
                  pygame.time.wait(2000)
                  self.reset()
                  break
              
          else:
              # Player 2
              positions = self.get_available_squares()
              player_2_action = 0
              if (self.player_2.player_type == 'human'):
                player_2_action = self.player_2.chooseAction_Human(positions)
              else:
                if (self.game_mode == "Random"):
                  print("random")
                  player_2_action = self.player_2.chooseAction_Random(positions)
                elif(self.game_mode == "RuleBased"):
                  print("RB")
                  player_2_action = self.player_2.chooseAction_RuleBased(positions, self.board)
                else:
                  print("RL")
                  player_2_action = self.player_2.chooseAction_QTable(positions, self.board)
              if (player_2_action!=-1):
                self.update_state(player_2_action)
                self.render()
                win = self.check_for_win()
                if win is not None:
                    pygame.time.wait(500)
                    if win == -1:
                        self.draw_win_text(self.player_2.name + " wins!")
                    else:
                        self.draw_win_text("Tie!")
                    pygame.time.wait(2000)
                    self.reset()
                    break
                
  