(window.webpackJsonp=window.webpackJsonp||[]).push([[8],{432:function(t,o,e){"use strict";e.r(o);var n=e(364),r=(e(76),{data:function(){return{board:[],width:160,height:210,player_1_position:96,player_2_position:96,ball_position:[96,80],ball_direction_x:-2,ball_direction_y:2,gameOver:!1,gameOverText:"",lastFourFrames:[],frame:0}},mounted:function(){var t=this;document.addEventListener("keyup",(function(o){"ArrowUp"===o.code?t.player_1_position-=10:"ArrowDown"===o.code&&(t.player_1_position+=10)})),this.resetGame(),setInterval((function(){t.renderFrame()}),0)},methods:{randomChoice:function(t){return t[Math.floor(t.length*Math.random())]},resetGame:function(){this.player_1_position=96,this.player_2_position=96,this.ball_position=[96,80],this.ball_direction_x=this.randomChoice([-2,2]),this.ball_direction_y=this.randomChoice([-2,2]),this.lastFourFrames=[]},renderFrame:function(){for(var t=[],i=0;i<this.height;i++){for(var o=[],e=0;e<this.width;e++)o.push([144,72,17]);t.push(o)}for(i=this.player_1_position;i<this.player_1_position+16;i++)for(e=16;e<20;e++)t[i][e]=[213,130,74];for(i=this.player_2_position;i<this.player_2_position+16;i++)for(e=140;e<144;e++)t[i][e]=[92,186,92];this.ball_position=[this.ball_position[0]+this.ball_direction_y,this.ball_position[1]+this.ball_direction_x];for(i=this.ball_position[0];i<this.ball_position[0]+4;i++)for(e=this.ball_position[1];e<this.ball_position[1]+2;e++)t[i][e]=[236,236,236];for(e=0;e<this.width;e++){for(i=194;i<this.height;i++)t[i][e]=[236,236,236];for(i=24;i<34;i++)t[i][e]=[236,236,236]}this.board=t;var n=this.ball_position[0],r=this.ball_position[1],l=this.player_1_position,h=this.player_1_position+16;n+4>=l&&n<=h&&(console.log("sw222itvh"),r>16&&r<=20&&(this.ball_direction_x=-1*this.ball_direction_x,console.log("switvh"))),l=this.player_2_position,h=this.player_2_position+16;n+4>=l&&n<=h&&r>=140&&r<=144&&(this.ball_direction_x=-1*this.ball_direction_x),n<=34&&(this.ball_direction_y=-1*this.ball_direction_y),n+4>=194&&(this.ball_direction_y=-1*this.ball_direction_y),(r<=0||r+2>=this.width)&&this.resetGame(),this.getAIMove(),this.renderGame()},getAIMove:function(){var t=this,o={frame:this.board};this.$store.dispatch("pingpong/getBestAction",o).then((function(o){3==o?t.player_2_position+16+5<194&&(t.player_2_position+=5):2==o&&t.player_2_position-5>34&&(t.player_2_position-=5)}))},renderGame:function(){for(var t,g,b,o=document.getElementById("myCanvas").getContext("2d"),e=this.board,i=0;i<e.length;i++)for(var n=0;n<e[0].length;n++)t=e[i][n][0],g=e[i][n][1],b=e[i][n][2],o.fillStyle="rgba("+t+","+g+","+b+", 1)",o.fillRect(n,i,1,1)}}}),l=e(78),component=Object(l.a)(r,(function(){var t=this,o=t._self._c;return o("div",{staticClass:"tictactoe-board game-window"},[o(n.a,{staticClass:"my-5 mx-auto",attrs:{flat:""}},[o("canvas",{attrs:{id:"myCanvas",width:t.width,height:t.height}})])],1)}),[],!1,null,"32440a3e",null);o.default=component.exports}}]);