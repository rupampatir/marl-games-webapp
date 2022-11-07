import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const debug = process.env.NODE_ENV != 'production'

const actions = {
//   async nuxtClientInit ({ commit, dispatch }) {
//     if (localStorage && localStorage.getItem('podium-third-party-cookies-disabled')) { localStorage.removeItem('podium-third-party-cookies-disabled') }
//     if (localStorage && localStorage.getItem('podium_cookie')) {
//       await commit('auth/login', {
//         auth: {
//           token: localStorage.getItem('podium_cookie')
//         }
//       })
//     }
//   }
}
const strict = debug

export default {
  actions,
  strict
}