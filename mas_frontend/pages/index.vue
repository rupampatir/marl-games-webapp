<template>
  <v-row justify="center" align="center">
    <v-col cols="12" sm="8" md="6">
      <v-card color="orange" class="logo mt-4 py-4 d-flex justify-center">
        <!-- <NuxtLogo />
        <VuetifyLogo /> -->
        <v-card-title class="h1 white--text">  <h1> Game Library </h1> </v-card-title>
      </v-card>

      
      <v-layout pt-9 class="mx-n4" wrap>
        <v-flex pb-6 xs6>
          <v-card class="pa-4 mx-4" color="white" max-width="300" hover>
            <v-img class="mb-6" contain max-height="150" src="tic-tac-toe.png" />
            <v-card-actions >
              <v-btn
                
                block
                color="orange"
                to="/tictactoe"
                nuxt
              >
                Play
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-flex>
        <v-flex pb-6 xs6>
          <v-card class="pa-4 mx-4" color="white" max-width="300" hover>
            <v-img class="mb-6" contain max-height="150" src="connect-4.png" />
            <v-card-actions >
              <v-btn
                
                block
                color="orange"
                to="/connect4"
                nuxt
              >
                Play
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-flex>
      <v-flex pb-6 xs6>
        <v-card class="pa-4 mx-4" color="white" max-width="300" hover>
          <v-img class="mb-6" contain max-height="150" src="snake.png" />
          <v-card-actions >
            <v-btn
              
              block
              color="orange"
              to="/snake"
              nuxt
            >
              Play
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-flex>

      </v-layout>
     
    </v-col>
  </v-row>
</template>

<script>
export default {
  name: 'MAS'
}
</script>
