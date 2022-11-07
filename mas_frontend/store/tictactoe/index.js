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
    getBestAction(context, payload) {
      return new Promise((resolve, reject) => {
        this.$axios.post('/get_tic_tac_toe_action/', payload).then(
          (response) => {
              console.log(response)
              resolve(response.action)
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