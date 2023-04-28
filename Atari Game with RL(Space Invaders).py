
## Import the packages that are required
import gym #package for Open AI gym
import random # random package to test randomness

## Create environment for Space invaders using open gym
env=gym.make('SpaceInvaders-v0') #Frame based environment returns an image as part of state

## Getting the values of height,width,channels from the environment.

height,width,channels=env.observation_space.shape
## Getting the action from env
actions=env.action_space.n #nUMBER OF ACTIONS THAT WE CAN TAKE

episodes=5 #Setup 5 episodes which means we play 5 games of space invaders

for episode in range(episodes):
    state=env.reset() # Reset the environment for every episode
    done=False
    score=0
    
    while not done:
        env.render()
        action=random.choice([0,1,2,3,4,5]) #We are taking random choice of 6 actions and perform them
        n_state,reward,done,info=env.step(action)
        score+=reward #We append rewards of each episode to variable score
    print(f"Episode:{episode}, Score {score}") #Prints the episode number and score in that episode
env.close()


