(window.webpackJsonp=window.webpackJsonp||[]).push([[6],{370:function(t,e,r){var content=r(407);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,r(67).default)("593d69fb",content,!0,{sourceMap:!1})},406:function(t,e,r){"use strict";r(370)},407:function(t,e,r){var o=r(66)(!1);o.push([t.i,".game-window[data-v-98a4c6b8]{max-width:40vw;margin:0 auto;padding-top:20px}.board[data-v-98a4c6b8]{background-color:#fff;padding:10px}.cell[data-v-98a4c6b8]{height:80px;width:80px;margin:4px;padding:2px;cursor:pointer;position:relative;border:3px solid #1f1f1f;border-radius:5px;background-color:#005f1e;transition:.2s}.cell.hover[data-v-98a4c6b8]{background-color:#ffae00;transform:scale(1.01)}.blue-chip[data-v-98a4c6b8]{background-color:#1500b5}.blue-chip[data-v-98a4c6b8],.red-chip[data-v-98a4c6b8]{height:50px;width:50px;border:2px solid #000;border-radius:50%;top:10%;left:10%;display:inline-block;z-index:3}.red-chip[data-v-98a4c6b8]{background-color:#b50000}.white-chip[data-v-98a4c6b8]{height:50px;width:50px;border:2px solid #000;background-color:#fff;border-radius:50%;top:10%;left:10%;display:inline-block;z-index:3}@media screen and (max-width:600px){.game-window[data-v-98a4c6b8]{max-width:90vw}}",""]),t.exports=o},431:function(t,e,r){"use strict";r.r(e);var o=r(388),n=r(364),c=r(351),d=r(430),l=r(369),h=r(425),v=(r(33),{data:function(){return{hover:-1,board:[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],PLAYER_PIECE:1,AI_PIECE:-1,currentPlayer:1,gameOver:!1,gameOverText:"",start_player:0}},computed:{humans_turn:function(){return 2==this.start_player&&-1==this.currentPlayer||1==this.start_player&&1==this.currentPlayer}},mounted:function(){},methods:{onMouseHover:function(col){this.currentPlayer==this.PLAYER_PIECE&&this.is_valid_location(col)&&(this.hover=col)},resetGame:function(){this.board=[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],this.currentPlayer=-1*this.currentPlayer,this.gameOver=!1,this.gameOverText="",this.currentPlayer==this.AI_PIECE&&this.getAIMove()},drop_piece:function(t,col,e){this.board[t][col]=e},is_valid_location:function(col){return 0==this.board[5][col]},get_next_open_row:function(col){for(var t=0;t<6;t++)if(0==this.board[t][col])return t},getAIMove:function(){var t=this,e={board:this.board};this.$store.dispatch("connect4/getBestAction",e).then((function(e){if(e.winner==t.PLAYER_PIECE)return t.gameOver=!0,void(t.gameOverText="You Win!");console.log("wololo",e);var r=t.get_next_open_row(e.action);t.drop_piece(r,e.action,t.AI_PIECE),t.$forceUpdate(),e.winner==t.AI_PIECE&&(t.gameOver=!0,t.gameOverText="You lose!"),t.currentPlayer=t.PLAYER_PIECE}))},performMove:function(t){if(this.is_valid_location(t)&&this.currentPlayer!=this.AI_PIECE){var e=this.get_next_open_row(t);this.drop_piece(e,t,this.PLAYER_PIECE),this.currentPlayer=this.AI_PIECE,this.getAIMove(),console.log("done"),this.hover=-1,this.$forceUpdate()}}}}),f=(r(406),r(78)),component=Object(f.a)(v,(function(){var t=this,e=t._self._c;return e("div",{staticClass:"tictactoe-board game-window"},[e(n.a,{staticClass:"pa-6",attrs:{color:"secondary",width:"100%"}},[e(l.a,{attrs:{column:"","justify-center":"","align-center":""}},[e("h2",{staticClass:"white--text"},[t._v(" "+t._s(t.humans_turn?"Your":"Computer's")+" Turn ")])])],1),t._v(" "),e(n.a,{staticClass:"board my-5 mx-auto",attrs:{flat:""}},t._l(6,(function(r,i){return e(l.a,{key:i,attrs:{"ma-n1":"","pa-0":""}},t._l(7,(function(r,o){return e(l.a,{key:"".concat(i).concat(o)},[e(l.a,{class:"cell ".concat(t.hover==o?"hover":""),on:{mouseover:function(e){return t.onMouseHover(o)},mouseleave:function(e){return t.onMouseHover(-1)},click:function(e){return t.performMove(o)}}},[e("div",{class:"".concat(0==t.board[5-i][o]?"white-chip":""," ").concat(-1==t.board[5-i][o]?"blue-chip":""," ").concat(1==t.board[5-i][o]?"red-chip":"")})]),t._v(" "),e("br")],1)})),1)})),1),t._v(" "),e(d.a,{attrs:{persistent:"","max-width":"290"},model:{value:t.gameOver,callback:function(e){t.gameOver=e},expression:"gameOver"}},[e(n.a,{staticClass:"pa-3",attrs:{color:"secondary"}},[e(c.b,{staticClass:"text-h3 text-center justify-center"},[t._v("\n        "+t._s(t.gameOverText)+"\n      ")]),t._v(" "),e(c.a,[e(h.a),t._v(" "),e(o.a,{attrs:{color:"black",block:""},on:{click:t.resetGame}},[t._v("\n          Restart\n        ")])],1)],1)],1)],1)}),[],!1,null,"98a4c6b8",null);e.default=component.exports}}]);