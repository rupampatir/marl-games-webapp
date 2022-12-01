<template>
  <div class="tictactoe-board game-window">
    <v-card class="pa-6" color="secondary" width="100%">
      <v-layout column justify-center align-center>
        <h2 class="white--text"> {{ humans_turn ? 'Your' : 'Computer\'s' }} Turn </h2>

      </v-layout>

    </v-card>
    <v-card flat class="my-5 mx-auto">
      <v-layout ma-n1 pa-0 v-for="(n, i) in 3" :key="i">
        <v-layout v-for="(n, j) in 3" :key="`${i}${j}`">
          <v-layout @click="performMove(i, j)" class="cell ma-1">
            <div :class="`${board[i][j] == -1 ? 'cross' : ''} ${board[i][j] == 1 ? 'dot' : ''}`"
              v-if="board[i][j] !== 0">
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
      board: [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ],
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
    this.start_player = Math.floor(Math.random() * 2) + 1
    if (this.start_player == 2) this.getAIMove()
  },
  methods: {
    resetGame() {
      this.board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ]
      this.start_player = Math.floor(Math.random() * 2) + 1
      this.currentPlayer = 1
      this.gameOver = false
      this.gameOverText = ''
      if (this.start_player == 2) this.getAIMove()
    },
    isGameOver() {
      return this.getPossibleMoves().length === 0 || this.playerHas3InARow(1) || this.playerHas3InARow(-1);
    },
    getPossibleMoves() {
      let moves = [];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (this.board[i][j] === 0) {
            moves.push({ x: i, y: j });
          }
        }
      }
      return moves;
    },

    getAIMove() {
      let payload = {
        board: this.board,
        start_player: this.start_player
      }
      console.log(payload)
      this.$store.dispatch('tictactoe/getBestAction', payload).then((action) => {
        this.board[action[0]][action[1]] = this.currentPlayer;
        this.$forceUpdate();


        if (this.isGameOver()) {
          this.gameOver = true;
          this.gameOverText = this.playerHas3InARow(this.currentPlayer) ? 'You lose!' : 'Draw';
          return;
        }

        this.currentPlayer = this.currentPlayer == 1 ? -1 : 1;

      });
    },

    playerHas3InARow(player) {
      // rows and cols
      for (let i = 0; i < 3; i++) {
        if (this.board[0][i] == player && this.board[1][i] == player && this.board[2][i] == player) {
          return true;
        }
        if (this.board[i][0] == player && this.board[i][1] == player && this.board[i][2] == player) {
          return true;
        }
      }

      // Diagonals
      if (this.board[0][0] == player && this.board[1][1] == player && this.board[2][2] == player) {
        return true;
      }
      if (this.board[0][2] == player && this.board[1][1] == player && this.board[2][0] == player) {
        return true;
      }

      return false;
    },
    performMove(x, y) {
      if (this.gameOver || !this.humans_turn) return
      if (this.board[x][y] != '') return
      this.board[x][y] = this.currentPlayer;

      this.$forceUpdate();

      if (this.isGameOver()) {
        this.gameOver = true;
        this.gameOverText = this.playerHas3InARow(this.currentPlayer) ? 'You win!' : 'Draw';
        return;
      }
      this.currentPlayer = this.currentPlayer == 1 ? -1 : 1;
      setTimeout(() => {
        this.getAIMove()
      }, 1000)

    }
  }
}
</script>


<style scoped>
.game-window {
  max-width: 35vw;
  margin: 0 auto;
  padding-top: 20px;
  

}

.cell {
  padding-bottom: 100%;
  cursor: pointer;
  /* width:1000px; */
  position: relative;
  border: solid 3px #1F1F1F;
  border-radius: 5px;
  background-color: #FFFF8F;
  -webkit-transition: 0.2s;
          transition: 0.2s;
}

.cell:hover {
  -webkit-transform: scale(1.05);
          transform: scale(1.05);
}

.cross {
  /* background: teal; */
  width: 80%;
  height: 80%;
  position: absolute;
  top: 10%;
  left: 40%;
}

.cross:after {
  content: '';
  height: 100%;
  border-left: 16px solid #76C49C;
  position: absolute;
  transform: rotate(45deg);
  /* left: 10%;
      top:10%; */
}

.cross:before {
  content: '';
  height: 100%;
  border-left: 16px solid #76C49C;
  position: absolute;
  transform: rotate(-45deg);
  /* top:10%;
      left: 10%; */
}

.dot {
  position: absolute;
  height: 80%;
  width: 80%;
  border: solid #89CFF0 16px;
  border-radius: 50%;
  top: 10%;
  left: 10%;
  display: inline-block;
  z-index: 3;
}

@media screen and (max-width: 600px) {
  .game-window {
    max-width: 90vw;
  }
}
</style>