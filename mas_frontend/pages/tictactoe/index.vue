<template>
  <div class="tictactoe-board game-window">
    <v-card class="pa-6" color="secondary" width="100%">
      <v-layout column justify-center align-center>
        <h1 class="text-xs-center white--text" v-if="gameOver">  {{ gameOverText }} </h1>
        <h2 class="white--text" v-else> {{currentPlayer==1?'Your':'Computer\'s'}} Turn </h2>
        <v-btn v-if="gameOver" class="mt-4" color="black" @click="resetGame">Restart</v-btn>
      </v-layout>
      
    </v-card>
    <v-card flat class="my-5 mx-auto" >
      <v-layout ma-n1 pa-0 v-for="(n, i) in 3" :key="i">
      <v-layout v-for="(n, j) in 3" :key="`${i}${j}`">
          <v-layout @click="performMove(i, j)" class="cell ma-1">
            <div :class="`${board[i][j]==1?'cross':''} ${board[i][j]==2?'dot':''}`" v-if="board[i][j] !== 0">
              <!-- {{ board[i][j]==1?'X':'O' }} -->
            </div>
          </v-layout>
          <br>
        </v-layout>
    </v-layout>
    </v-card>
   
    <div class="game-over-text" v-if="gameOver">
     
    </div>
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
      currentPlayer: 0,
      gameOver: false,
      gameOverText: '',
      start_player: 0
    }
  },
  computed: {
    
  },
  mounted() {
    this.start_player = Math.floor(Math.random() * 2) + 1
    console.log(this.start_player)
    if (this.start_player==2) this.getAIMove()
    this.currentPlayer = this.start_player
  },
  methods: {
    resetGame() {
      this.board = [
      [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ]
      this.start_player = Math.floor(Math.random() * 2) + 1
      this.currentPlayer = this.start_player
      this.gameOver = false
      this.gameOverText = ''
      if (this.start_player==2) this.getAIMove()
    },
    isGameOver() {
      return this.getPossibleMoves().length === 0 || this.playerHas3InARow(1) || this.playerHas3InARow(2);
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
    getScore() {
      let score = 0;
      if (this.playerHas3InARow(1)) {
        score -= 100;
      }
      if (this.playerHas3InARow(2)) {
        score += 100;
      }
      return score;
    },

    getAIMove() {
      let payload = {
        board:JSON.stringify(this.board.flat().join("")), 
        start_player: this.start_player
      }
      this.$store.dispatch('tictactoe/getBestAction', payload).then((action) => {
        this.board[action[0]][action[1]] = this.currentPlayer;
        this.$forceUpdate();


        if (this.isGameOver()) {
          this.gameOver = true;
          this.gameOverText = this.playerHas3InARow(this.currentPlayer) ? 'You lose!' : 'Draw';
          return;
        }

        this.currentPlayer = this.currentPlayer == 1 ? 2 : 1;
        
      });
    },

    playerHas3InARow(player) {
      // Horizontal rows
      for (let i = 0; i < 3; i++) {
        if (this.board[0][i] === player && this.board[1][i] === player && this.board[2][i] === player) {
          return true;
        }
      }

      // Vertical rows
      for (let i = 0; i < 3; i++) {
        if (this.board[i][0] === player && this.board[i][1] === player && this.board[i][2] === player) {
          return true;
        }
      }

      // Diagonals
      if (this.board[0][0] === player && this.board[1][1] === player && this.board[2][2] === player) {
        return true;
      }
      if (this.board[1][0] === player && this.board[1][1] === player && this.board[0][2] === player) {
        return true;
      }

      return false;
    },
    performMove(x, y) {
      if (this.gameOver) return
      if (this.board[x][y] != '') return
      this.board[x][y] = this.currentPlayer;

      this.$forceUpdate();

      if (this.isGameOver()) {
        this.gameOver = true;
        this.gameOverText = this.playerHas3InARow(this.currentPlayer) ? 'You win!' : 'Draw';
        return;
      }
      this.currentPlayer = this.currentPlayer == 1 ? 2 : 1;
      setTimeout(() => {
        this.getAIMove()
      },1000)
      
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
    padding-bottom:100%;
    /* width:1000px; */
    position: relative;
    border: solid 3px #1F1F1F;
    border-radius: 5px;
    background-color: #FFFF8F;
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
    top:10%;
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