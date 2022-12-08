<template>
  <div class="tictactoe-board game-window">
    <v-card class="pa-6" color="secondary" width="100%">
      <v-layout column justify-center align-center>
        <h2 class="white--text"> {{ humans_turn ? 'Your' : 'Computer\'s' }} Turn </h2>
      </v-layout>

    </v-card>

    <v-card flat class="board my-5 mx-auto">
      <v-layout ma-n1 pa-0 v-for="(n, i) in 6" :key="i">
        <v-layout v-for="(n, j) in 7" :key="`${i}${j}`">
          <v-layout  
            @mouseover="onMouseHover(j)"
            @mouseleave="onMouseHover(-1)" 
            @click="performMove(j)" :class="`cell ${hover==j?'hover':'invalid-hover'}`">
            <div :class="`${board[5-i][j] == 0 ? 'white-chip' : ''} ${board[5-i][j] == -1 ? 'blue-chip' : ''} ${board[5-i][j] == 1 ? 'red-chip' : ''}`">
            </div>
          </v-layout>
          <br>
        </v-layout>
      </v-layout>
    </v-card>

    <v-dialog v-model="gameOver" persistent max-width="290">

      <v-card class="pa-3" color="secondary" >

        <v-card-title class="text-h3 text-center justify-center">
          {{ gameOverText }}
        </v-card-title>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="black" block @click="resetGame">
            Restart
          </v-btn>

        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
<script>

export default {
  data() {
    return {
      hover: -1,
      board: [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
      ],
      PLAYER_PIECE: 1,
      AI_PIECE : -1,
      currentPlayer: 1,
      gameOver: false,
      gameOverText: '',
      start_player: 0,
    }
  },
  computed: {
    humans_turn() {
      return (this.start_player == 2 && this.currentPlayer == -1) || (this.start_player == 1 && this.currentPlayer == 1)
    },
  },
  mounted() {
  },
  methods: {
    onMouseHover(col) {
      if (this.currentPlayer==this.PLAYER_PIECE && this.is_valid_location(col)) this.hover = col
    },
    resetGame() {
      this.board = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
      ]
      this.currentPlayer = this.currentPlayer*-1
      this.gameOver = false
      this.gameOverText = ''
      if (this.currentPlayer == this.AI_PIECE) this.getAIMove()
    },
    drop_piece(row, col, piece) {
      this.board[row][col] = piece
    },

    is_valid_location(col) {
      return this.board[5][col] == 0
    },

    get_next_open_row(col) {
      for (let r=0;r<6;r++) {
        if (this.board[r][col] == 0) return r
      } 
    },
    getAIMove() {
      
      let payload = {
        board: this.board
      }
      this.$store.dispatch('connect4/getBestAction', payload).then((res) => {
        if (res.winner==this.PLAYER_PIECE) {
          this.gameOver = true;
          this.gameOverText = 'You Win!'
          return;
        } else {
          console.log("wololo", res)
          let row = this.get_next_open_row(res.action)
          this.drop_piece(row, res.action, this.AI_PIECE)
          this.$forceUpdate();
          if (res.winner == this.AI_PIECE) {
            this.gameOver = true;
            this.gameOverText = 'You lose!'
          }
        }
        this.currentPlayer = this.PLAYER_PIECE
      });
    },

  
    performMove(action) {
      if (!this.is_valid_location(action) || this.currentPlayer==this.AI_PIECE) return
      let row = this.get_next_open_row(action)
      this.drop_piece(row, action, this.PLAYER_PIECE)
      this.currentPlayer = this.AI_PIECE
      this.getAIMove()
      console.log("done")
      this.hover = -1
      this.$forceUpdate();

    }
  }
}
</script>


<style scoped>
.game-window {
  max-width: 40vw;
  margin: 0 auto;
  padding-top: 20px;

}

.board {
  background-color: white;
  padding: 10px;
}

.cell {
  height: 80px;

  width: 80px;
  margin:4px;
  padding:2px;

  cursor: pointer;
  /* width:1000px; */
  position: relative;
  border: solid 3px #1F1F1F;
  border-radius: 5px;
  background-color: #005f1e;
  -webkit-transition: 0.2s;
          transition: 0.2s;
}

.cell.hover {
  background-color: #ffae00;

  -webkit-transform: scale(1.01);
          transform: scale(1.01);
}
.cell.invalid-hover {
  cursor: not-allowed;
}
.blue-chip {
  height: 50px;
  width: 50px;
  border: solid black 2px;
  background-color: rgb(21, 0, 181);
  border-radius: 50%;
  margin: auto;

  display: inline-block;
  z-index: 3;
}

.red-chip {
  height: 50px;
  width: 50px;
  border: solid black 2px;
  background-color: rgb(181, 0, 0);
  border-radius: 50%;
  margin: auto;

  display: inline-block;
  z-index: 3;
}

.white-chip {
  height: 50px;
  width: 50px;
  border: solid black 2px;
  background-color: white;
  margin: auto;
  border-radius: 50%;
  display: inline-block;
  z-index: 3;
}

@media screen and (max-width: 600px) {
  .game-window {
    max-width: 90vw;
  }
}
</style>