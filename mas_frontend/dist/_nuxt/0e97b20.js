(window.webpackJsonp=window.webpackJsonp||[]).push([[2],{157:function(t,n,o){"use strict";o(5);n.a=function(t,n){var o=t.$axios;o.onRequest((function(t){return t.headers={common:{"Content-Type":"application/json"}},t})),o.onResponse((function(t){return t.data})),o.onError((function(t){return Promise.resolve(t)})),n("api",{get:function(t,n){return new Promise((function(e,r){console.log("wololo",n),o.get(t,{params:n||{}}).then((function(data){e(data)}),(function(t){r(t)}))}))},post:function(t,n){return new Promise((function(e,r){o.post(t,n).then((function(data){e(data)}),(function(t){r(t)}))}))}})}},212:function(t,n,o){var content=o(298);content.__esModule&&(content=content.default),"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,o(67).default)("4d6a2e0f",content,!0,{sourceMap:!1})},233:function(t,n,o){"use strict";var e=o(348),r=o(349),c=o(350),f={name:"DefaultLayout",data:function(){return{clipped:!1,drawer:!1,fixed:!1,items:[{icon:"mdi-apps",title:"Welcome",to:"/"},{icon:"mdi-chart-bubble",title:"Inspire",to:"/inspire"}],miniVariant:!1,right:!0,rightDrawer:!1,title:"Vuetify.js"}}},l=o(78),component=Object(l.a)(f,(function(){var t=this._self._c;return t(e.a,{attrs:{dark:""}},[t(c.a,[t(r.a,[t("Nuxt")],1)],1)],1)}),[],!1,null,null,null);n.a=component.exports},244:function(t,n,o){o(245),t.exports=o(246)},297:function(t,n,o){"use strict";o(212)},298:function(t,n,o){var e=o(66)(!1);e.push([t.i,"h1[data-v-35e10596]{font-size:20px}",""]),t.exports=e},315:function(t,n,o){"use strict";o.r(n);var e=o(0),r=o(74);e.a.use(r.a);n.default={actions:{},strict:false}},316:function(t,n,o){"use strict";o.r(n);o(5);var e={getBestAction:function(t,n){var o=this;return new Promise((function(t,e){o.$axios.post("/get_tic_tac_toe_action/",n).then((function(n){console.log(n),t(n.action)}),(function(t){e(t)}))}))}};n.default={namespaced:!0,actions:e,mutations:{},state:function(){return{}}}},317:function(t,n,o){"use strict";o.r(n);o(5);var e={resetGame:function(t,n){var o=this;return new Promise((function(t,e){o.$axios.post("/reset_snake_game/",n).then((function(n){console.log(n),t(n)}),(function(t){e(t)}))}))},getNextState:function(t,n){var o=this;return new Promise((function(t,e){o.$axios.post("/get_snake_action/",n).then((function(n){console.log(n),t(n)}),(function(t){e(t)}))}))}};n.default={namespaced:!0,actions:e,mutations:{},state:function(){return{}}}},318:function(t,n,o){"use strict";o.r(n);o(5);var e={getBestAction:function(t,n){var o=this;return new Promise((function(t,e){o.$axios.post("/get_pong_action/",n).then((function(n){console.log(n),t(n.action)}),(function(t){e(t)}))}))}};n.default={namespaced:!0,actions:e,mutations:{},state:function(){return{}}}},319:function(t,n,o){"use strict";o.r(n);o(5);var e={getBestAction:function(t,n){var o=this;return new Promise((function(t,e){o.$axios.post("/get_connect_4_action/",n).then((function(n){console.log(n),t(n)}),(function(t){e(t)}))}))}};n.default={namespaced:!0,actions:e,mutations:{},state:function(){return{}}}},63:function(t,n,o){"use strict";var e=o(348),r={name:"EmptyLayout",layout:"empty",props:{error:{type:Object,default:null}},data:function(){return{pageNotFound:"404 Not Found",otherError:"An error occurred"}},head:function(){return{title:404===this.error.statusCode?this.pageNotFound:this.otherError}}},c=(o(297),o(78)),component=Object(c.a)(r,(function(){var t=this,n=t._self._c;return n(e.a,{attrs:{dark:""}},[404===t.error.statusCode?n("h1",[t._v("\n    "+t._s(t.pageNotFound)+"\n  ")]):n("h1",[t._v("\n    "+t._s(t.otherError)+"\n  ")]),t._v(" "),n("NuxtLink",{attrs:{to:"/"}},[t._v("\n    Home page\n  ")])],1)}),[],!1,null,"35e10596",null);n.a=component.exports}},[[244,11,3,12]]]);