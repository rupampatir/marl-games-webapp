<template>
    <div class="tictactoe-board game-window">
      <v-card class="pa-6" color="secondary" width="100%">
        <v-layout column justify-center align-center>
          <h2 class="white--text">  Score   </h2>
          <v-layout :style="`width:100%;`" justify-space-around>
            <h1 :style="`color: rgb(21, 0, 181);`"> YOU: {{ scores[1] }}</h1>
            <h1 :style="`color: rgb(181, 0, 0);`"> {{ scores[0] }} :AI</h1>

          </v-layout>

        </v-layout>
        <v-btn color="black" block @click="startGame">
              Start/Restart
        </v-btn>
      </v-card>
  
      <v-card flat class="snakeborder my-5 mx-auto">
        <v-layout pa-0 v-for="(n, i) in 20" :key="i">
          <div justify-center class="backface" v-for="(n, j) in 20" :key="`${i}${j}`">
              <div :class="`snake-cell  ${board[i][j] == 1 ? 'blue_snake' : ''} ${board[i][j] == 2 ? 'red_snake' : ''} ${board[i][j] == -1 ? 'food' : ''}`">
            <!-- {{board[i][j]}} -->
              </div>
          </div>
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
            // console.log(self.lastMovePerformed)

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
    /* max-width: 35vw; */
    margin: 0 auto;
    padding-top: 20px;
    
  
  }
  
  .snakeborder {
    width: fit-content;
    border: solid 14px grey !important;
    background-color: #FFFF8F;
  }
  
  .snake-cell {
    
    height: 20px;
    width: 20px;
    /* border: solid 1px #1F1F1F; */
    background-color: #FFFF8F;

    
    /* opacity: 0.3; */
    -webkit-transition: 0.01s;
            transition: 0.01s;
  }
  
  .snake-cell.red_snake {
    border: solid 2px black;
    border-radius: 50%;
    background-color: rgb(181, 0, 0);
  }
  
  .snake-cell.blue_snake {
    border: solid 2px black;
    border-radius: 50%;    
    background-color: rgb(21, 0, 181);
  }

  .snake-cell.food {
    background-image: url('/food.png');

  }
  
  @media screen and (max-width: 600px) {
    .game-window {
      max-width: 90vw;
    }
  }
  </style>