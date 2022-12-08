<template>
    <div class="tictactoe-board game-window">
      <v-card class="d-flex justify-space-between pa-6" color="secondary" width="100%">
        <v-flex xs9>
          <v-layout column justify-center align-center>
            <h1 class="black--text">  SCORE   </h1>
            <v-layout column :style="`width:100%;`" align-start>
              <h1 :style="`color: rgb(21, 0, 181);`"> You: {{ scores[0]?scores[0]:0 }}</h1>
              <h1 v-for="(n,i) in selectedNumberOfOpponents" :key="`opponent${i}`" :style="`color: ${color_mapping[i+2]};`"> AI {{i+1}}: {{ scores[i+1] ? scores[i+1] : 0 }}</h1>
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
                v-model="selectedNumberOfOpponents"
                :hint="`Number of opponents`"
                :items="[1,2,3]"
                persistent-hint
                outlined
                dense
                label="Number of opponents"
                single-line
              ></v-select>
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
          <div justify-center v-for="(n, j) in 20" :key="`${i}${j}`">
              <div v-if="board[i][j]==-1" class="snake_cell food" > </div>
              <div v-else-if="(board[i][j]!=0)" class="snake_cell player" :style="`background-color: ${color_mapping[board[i][j]]};`"></div>
              <div v-else class="snake_cell"></div>
            <!-- {{board[i][j]}} -->
          </div>
        </v-layout>
      </v-card>
  
      
    </div>
  </template>
  <script>
  
  export default {
    data() {
      return {
        color_mapping: {
          1: "rgb(21, 0, 181)",
          2: "rgb(181, 0, 0)",
          3: "rgb(0, 126, 19);",
          4: "rgb(238, 0, 255)"
        },
        hover: -1,
        board: [],
        scores: 0,
        pause: true,
        started: false,
        lastMovePerformed: 3,
        selectedNumberOfFood: 1,
        selectedNumberOfOpponents: 1
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
            "num_food": this.selectedNumberOfFood,
            "num_opponents": this.selectedNumberOfOpponents
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
  
  .snake_cell {
    
    height: 20px;
    width: 20px;
    /* border: solid 1px #FFFF8F; */
    background-color: #FFFF8F;
  
    
    /* opacity: 0.3; */
    -webkit-transition: 0.01s;
            transition: 0.01s;
  }

  .snake_cell.player {
    border: solid 2px black;
    border-radius: 50%;   
  }

  .snake_cell.food {
    background-image: url('/food.png');

  }
  
  @media screen and (max-width: 600px) {
    .game-window {
      max-width: 90vw;
    }
  }
  </style>