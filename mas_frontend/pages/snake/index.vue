<template>
    <div class="tictactoe-board game-window">
      <v-card class="d-flex justify-space-between pa-6" color="secondary" width="100%">
        <v-flex xs9>
          <v-layout column justify-center align-center>
            <h1 class="black--text">  SCORE   </h1>
            <v-layout :style="`width:100%;`" justify-space-around>
              <h1 :style="`color: rgb(21, 0, 181);`"> You: {{ scores[1] }}</h1>
              <h1 :style="`color: rgb(181, 0, 0);`"> {{ scores[0] }} :AI</h1>
            </v-layout>
          </v-layout>
        </v-flex>
        <v-flex xs3>

          <v-layout column v-if="started">
            <v-btn v-if="paused" color="black" block @click="continueGame">
                Resume 
            </v-btn>
            <v-btn v-else color="black"  @click="pauseGame">
                Pause 
            </v-btn>
            <v-btn class="mt-2"  color="black"  @click="stopGame">
                Restart
            </v-btn>
          </v-layout>
         
          <v-card color="black" class="pa-4" v-else> 
            <v-layout column>
            <v-btn color="green"  @click="startGame">
                 <span class="black--text"> Start</span>
            </v-btn>
            <v-select
                class="mt-2" 
                item-color="white"
                color="white"
                v-model="selectedNumberOfFood"
                :hint="`Number of apples`"
                :items="[1,2,3,4]"
                persistent-hint
                outlined
                dense
                label="Number of apples"
                single-line
              ></v-select>
          </v-layout>
          </v-card>
        </v-flex>
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
        pause: true,
        started: false,
        lastMovePerformed: 3,
        selectedNumberOfFood: 1
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
        this.started = true
        this.paused = false
        let payload = {
            "num_food": this.selectedNumberOfFood
        }
        this.$store.dispatch('snake/resetGame', payload).then((res) => {
            this.board = res.board
            this.scores = res.scores
            setTimeout(()=>this.runGame(),100)
        })
      },
      pauseGame() {
        this.paused = true
        this.$forceUpdate()
      },
      continueGame() {
        this.paused = false
        this.runGame()
      },
      stopGame() {
        this.started = false
        this.paused = true
      },
      runGame() {
        if (this.paused) return
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