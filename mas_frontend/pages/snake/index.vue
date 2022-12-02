<template>
    <div class="tictactoe-board game-window">
      <v-card class="pa-6" color="secondary" width="100%">
        <v-layout column justify-center align-center>
          <h2 class="white--text"> Max Score: {{ scores }}  </h2>
        </v-layout>
        <v-btn color="black" block @click="startGame">
              Start/Restart
        </v-btn>
      </v-card>
  
      <v-card flat class="my-5 mx-auto">
        <v-layout ma-n1 pa-0 v-for="(n, i) in 20" :key="i">
          <v-layout v-for="(n, j) in 20" :key="`${i}${j}`">
              <div :class="`cell ${board[i][j] == 1 ? 'blue' : ''} ${board[i][j] == 2 ? 'red' : ''} ${board[i][j] == -1 ? 'food' : ''}`">
              </div>
          </v-layout>
        </v-layout>
      </v-card>
  
      
    </div>
  </template>
  <script>
  
  export default {
    data() {
      return {
        hover: -1,
        board: [],
        scores: 0,
        lastMovePerformed: 3
      }
    },
    computed: {
      humans_turn() {
        return (this.start_player == 2 && this.currentPlayer == -1) || (this.start_player == 1 && this.currentPlayer == 1)
      },
    },
    beforeMount() {
        for (let i=0;i<20;i++) {
            let temp = []
            for (let j=0;j<20;j++) {
               temp.push(0)
            }
            this.board.push(temp)
        }
    },
    mounted() {
        
        let self = this
        function checkKey(e) {
            
            e = e || window.event;
            // self.right = 0
            // self.left = 1
            // self.up = 2
            // self.down = 3
            if (e.keyCode == '38') {
                self.lastMovePerformed = 1
            }
            else if (e.keyCode == '40') {
                // down arrow
                self.lastMovePerformed = 0
            }
            else if (e.keyCode == '37') {
            // left arrow
            self.lastMovePerformed = 2

            }
            else if (e.keyCode == '39') {
            // right arrow

            self.lastMovePerformed = 3

            }
            console.log(self.lastMovePerformed)

        }
        document.onkeydown = checkKey;

    },
    methods: {
      onMouseHover(col) {
        if (this.currentPlayer==this.PLAYER_PIECE && this.is_valid_location(col)) this.hover = col
      },
      startGame() {
        this.$store.dispatch('snake/resetGame').then((res) => {
            this.board = res.board
            this.scores = res.scores
            setTimeout(()=>this.runGame(),100)
        })
      },
      runGame() {
        let payload = {
            "action": this.lastMovePerformed
        }
        this.$store.dispatch('snake/getNextState', payload).then((res) => {
            this.board = res.board
            this.scores = res.scores
            setTimeout(()=>this.runGame(),100)
        })
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
    height: 30px;
    width: 30px;
    border: solid 1px #1F1F1F;
    background-color: #FFFF8F;
    -webkit-transition: 0.2s;
            transition: 0.2s;
  }
  
  .cell.red {
    background-color: red;
  }
  
  .cell.blue {
    background-color: blue;
  }

  .cell.food {
    background-color: black;
  }
  
  @media screen and (max-width: 600px) {
    .game-window {
      max-width: 90vw;
    }
  }
  </style>