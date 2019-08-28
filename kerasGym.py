#not an actual AI proj just Testing Soemthing
#ignore this
import gym
env = gym.make('CartPole-v0')
done = False
while done == False:
    env.reset()
    env.render()
    i = input()
    if i == 'a':
        action = 0
        observation, reward, done, info = env.step(action)
    else:
        action = 1
        observation, reward, done, info = env.step(action)


print('its done')
