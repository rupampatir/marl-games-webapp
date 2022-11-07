export default function ({ $axios }, inject) {
    $axios.onRequest((config) => {
    config.headers = {
        common: {
        // 'content-management-api-key': 'm5p2cz5qzc9zbfncq5avk2mkqwxkawskw3ehgsddjjtupn3vjhp8p95b8bst8mzw',
        'Content-Type': 'application/json',
        },
        
      }
    //   console.log(config)
      return config
    })
    $axios.onResponse((response) => {
      return response.data
    })
    $axios.onError((error) => {
      return Promise.resolve(error)
    })
  
    const api = {
      get(url, requestData) {
        return new Promise((resolve, reject) => {
            console.log("wololo",requestData)
          $axios.get(url, { params: requestData || {} }).then(
            (data) => {
              resolve(data)
            },
            (err) => {
              reject(err)
            }
          )
        })
      },
      post(url, requestData) {
        return new Promise((resolve, reject) => {
          $axios.post(url, requestData).then(
            (data) => {
              resolve(data)
            },
            (err) => {
              reject(err)
            }
          )
        })
      },
    }
  
    inject('api', api)
  }