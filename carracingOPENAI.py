import gym
import numpy as np
import tflearn
import  random
from tflearn.layers.core import fully_connected,input_data,dropout
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
env.reset()

def some_random_games_first():
    for _ in range(1):
        env.reset()
        for o in range(200):
            print('o is '+ str(o))
            env.render()
            for p in range(100):
                if p<50:
                    action = 0
                else:
                    action = 2
                if p == 99:
                    p = 0
                print(str(action))
                observation, reward, done, info = env.step(action)
                if reward == 1:
                    print('yoloo')
                if done:
                    break
#some_random_games_first()
def initial_game_data():
    training_data = []
    scores = []
    accepted_score = []
    for _ in range(5):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(100):
            action = [random.uniform(-1.99,1.99)]
            observation, reward, done, info = env.step(action)
            score = score + reward
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            if done:
                break

        if score > -500:
            accepted_score.append(score)
            for data in game_memory:
                if 0 < data[1][0] < 2:
                    output = [0, 1]
                elif -2 < data[1][0] < 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        print(scores)
        env.reset()
        scores.append(score)
    #print('Average accepted score:', mean(accepted_score))
    #print('Median score for accepted scores:', median(accepted_score))
    #print(Counter(accepted_score))

    return training_data

def neural_network(input_size):
    network = input_data(shape=[None,input_size, 1], name='input')

    network = fully_connected(network, 128 ,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network , 2 ,activation='softmax')
    network =  regression(network,optimizer='adam',loss='categorical_crossentropy',name='targets')
    model = tflearn.DNN(network)
    return model
def train_model(training_data,model = False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]), 1)
    y = y = [i[1] for i in training_data]
    if not model:
        model = neural_network(input_size=len(X[0]))


    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data = initial_game_data()
model = train_model(training_data)

scores = []
choices = []
for each_game in range(5):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(500):
        env.render()

        if len(prev_obs) == 0:
            action = [random.uniform(-1.99,1.99)]
        else:
            action = [np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])]

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done: break

    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print('-500')




