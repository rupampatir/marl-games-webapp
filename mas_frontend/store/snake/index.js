const state = () => ({
    
  })
  
  const mutations = {
    // login: (state, payload) => {
    //   localStorage.setItem('podium_backoffice_user', JSON.stringify(payload))
    //   state.isLoggedIn = true
    //   state.user = payload
    // },
  }
  
  const actions = {
    resetGame(context, payload) {
      return new Promise((resolve, reject) => {
        this.$axios.post('/reset_snake_game/', payload).then(
          (response) => {
              console.log(response)
              resolve(response)
          },
          (err) => {
            reject(err)
          }
        )
      })
    },
    getNextState(context, payload) {
      return new Promise((resolve, reject) => {
        this.$axios.post('/get_snake_action/', payload).then(
          (response) => {
              console.log(response)
              resolve(response)
          },
          (err) => {
            reject(err)
          }
        )
      })
    },
  }
  
  export default {
    namespaced: true,
    actions,
    mutations,
    state,
  }