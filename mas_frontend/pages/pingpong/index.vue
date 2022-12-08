<template>
    <div class="tictactoe-board game-window">
      <!-- <v-card class="pa-6" color="secondary" width="100%">
        <v-layout column justify-center align-center>
          <h2 class="white--text"> {{ humans_turn ? 'Your' : 'Computer\'s' }} Turn </h2>
  
        </v-layout> -->
  
      <!-- </v-card> -->
      <v-card flat class="my-5 mx-auto">
        <canvas id="myCanvas" :width="width" :height="height"></canvas> <!-- change with real sizes --> 
      </v-card>
  
      <!-- <v-dialog v-model="gameOver" persistent max-width="290">
  
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
      </v-dialog> -->
    </div>
  </template>
  <script>
  
  export default {
    data() {
      return {
        board: [],

        width: 160,
        height: 210,
        
        player_1_position: 96,
        player_2_position: 96,    
        ball_position: [96,80],
        ball_direction_x: -2,
        ball_direction_y: 2,
        gameOver: false,
        gameOverText: '',
        lastFourFrames: [],
        frame: 0
      }
    },
    // computed: {
    //   humans_turn() {
    //     return (this.start_player == 2 && this.currentPlayer == -1) || (this.start_player == 1 && this.currentPlayer == 1)
    //   },
    // },
    mounted() {
        document.addEventListener('keyup', (e) => {
            if (e.code === "ArrowUp")        this.player_1_position -= 10
            else if (e.code === "ArrowDown") this.player_1_position += 10
        });
        this.resetGame()
        // this.renderFrame()
        // this.renderFrame()
        // this.renderFrame()
        setInterval(() => {
            this.renderFrame()

            }, 0)
    },
    methods: {
      randomChoice(arr) {
          return arr[Math.floor(arr.length * Math.random())];
      },
        resetGame() {
            this.player_1_position = 96;
            this.player_2_position = 96;    
            this.ball_position = [96,80];
            this.ball_direction_x = this.randomChoice([-2,2]);
            this.ball_direction_y = this.randomChoice([-2,2]);
            this.lastFourFrames = []
        },
      renderFrame() {
        let board = []
        for(var i=0; i< this.height; i++){ 
            let row = []
            for(var j=0; j<  this.width; j++){ 
                row.push([144,72,17])
            }
            board.push(row)
        }
        
        // set paddles
        const paddleWidth = 4;
        const paddleHeight = 16;
        const player1Offset = 16;
        const player2Offset = 140;

        // Player 1 paddle
        for (var i=this.player_1_position; i<this.player_1_position+paddleHeight; i++) {
            for(var j=player1Offset; j< player1Offset+paddleWidth; j++){ 
                board[i][j] = [213,130,74]
            }
        }
        
        // Player 2 paddle
         for (var i=this.player_2_position; i<this.player_2_position+paddleHeight; i++) {
            for(var j=player2Offset; j< player2Offset+paddleWidth; j++){ 
                board[i][j] = [92,186,92]
            }
        }

        // ball
        const ballWidth = 2;
        const ballHeight = 4;
        this.ball_position = [this.ball_position[0] + this.ball_direction_y, this.ball_position[1] + this.ball_direction_x]
        // console.log(this.ball_position)
        for (var i=this.ball_position[0]; i<this.ball_position[0]+ballHeight; i++) {
            for(var j=this.ball_position[1]; j< this.ball_position[1]+ballWidth; j++){ 
                board[i][j] = [236,236,236]
            }
        }
        // Player 2 paddle
        // Ball 1: 28,165
        //    
        // Ball 2: 27, 166
        // Ball 2: 26, 167

        // Ball 3: 25, 168

        // Player 1/1: 165
        // Player 1/2: 165
        // Player 1/3: 167
        // Player 1/4: 167
        


        // player 2/1: 126
        // player 2/2: 136
        // player 2/3: 136
        // player 2/4: 148


        for(var j=0; j< this.width; j++){ 
            for (var i=194; i<this.height; i++) {
                board[i][j] = [236,236,236]
            }
            for (var i=24; i<34; i++) {
                board[i][j] = [236,236,236]
            }
        }
        this.board = board
        
        let ball_y = this.ball_position[0]
        let ball_x = this.ball_position[1]
        // if it hit the paddle on left
        let paddle_top = this.player_1_position
        let paddle_bottom = this.player_1_position+paddleHeight
        let paddle_left = player1Offset + paddleWidth
        // console.log(ball_y+ballHeight,paddle_top)
        if (ball_y+ballHeight >= paddle_top && ball_y<=paddle_bottom) {
            console.log('sw222itvh')

            if (ball_x>player1Offset && ball_x<=paddle_left) {
                this.ball_direction_x=this.ball_direction_x*-1
                console.log('switvh')
                // var relativeIntersectY = (paddle1Y+(PADDLEHEIGHT/2)) - intersectY;





            }
        }


        // if it hit the paddle on right
        paddle_top = this.player_2_position
        paddle_bottom = this.player_2_position+paddleHeight
        let paddle_right = player2Offset
        if (ball_y+ballHeight>=paddle_top && ball_y<=paddle_bottom) {
            if (ball_x>=paddle_right && ball_x<=paddle_right+paddleWidth) {
                this.ball_direction_x=this.ball_direction_x*-1
            }
        }

        // if it hit top screen
        if (ball_y<=34) {
            this.ball_direction_y=this.ball_direction_y*-1
        }
        if (ball_y+ballHeight>=194) {
            this.ball_direction_y=this.ball_direction_y*-1
        }
        if (ball_x<=0 || ball_x+ballWidth>=this.width) {
            // this.gameOver()
            this.resetGame()

        }
        // if (this.lastFourFrames.length>=4) {
        //     // if (this.frame%4==0) {
        //         this.lastFourFrames.shift()
        //         this.lastFourFrames.push(JSON.parse(JSON.stringify(board))) 
        //     // }
        //     this.frame++
          
        this.getAIMove()
        // } else {
        //     this.lastFourFrames.push(JSON.parse(JSON.stringify(board))) 
        // }
        this.renderGame()

      },
      getAIMove() {
        let payload = {
            frame: this.board,
        }
        this.$store.dispatch('pingpong/getBestAction', payload).then((action) => {
            // this.board[action[0]][action[1]] = this.currentPlayer;
            if (action==3 ) {
                if (this.player_2_position+16+5<194)
                    this.player_2_position += 5
            } else if (action==2) {
                if (this.player_2_position-5>34)
                    this.player_2_position -= 5
            }
            
        })
      },
      renderGame() {
        var c = document.getElementById("myCanvas"); 
        var ctx = c.getContext("2d"); 

        // draw board
        const rgbdata = this.board
        var r,g,b; 
        for(var i=0; i< rgbdata.length; i++){ 
            for(var j=0; j< rgbdata[0].length; j++){ 
                r = rgbdata[i][j][0]; 
                g = rgbdata[i][j][1];	 
                b = rgbdata[i][j][2];		 
                ctx.fillStyle = "rgba("+r+","+g+","+b+", 1)";  
                ctx.fillRect( j, i, 1, 1 ); 
            } 
        } 

       
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
    width: 2px;
    height: 2px;
  }
  