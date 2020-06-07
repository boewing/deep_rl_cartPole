from datetime import datetime
import time 
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from collections import defaultdict

# import tensorflow.keras as keras
# from tensorflow.keras import layers, Model
# from tensorflow.keras.models import model_from_json

import keras 
from keras import layers, Model
from keras.models import model_from_json

def dense_nn(input_dim, layers_sizes, scope_name):
    """Creates a densely connected multi-layer neural network.
    inputs: the input tensor
    layers_sizes (list<int>): defines the number of units in each layer. The output 
        layer has the size layers_sizes[-1].
    """
    model = keras.Sequential()
    for layer_size in layers_sizes[:-1]:
        model.add(layers.Dense(layer_size, input_dim=input_dim, activation=keras.activations.relu))
        input_dim = None
    model.add(layers.Dense(layers_sizes[-1], input_dim=input_dim, activation=keras.activations.linear))
    return model


def update_Q(Q : Model, remember):
    global experiments_left
    global old_q_vals_to_fit
    global s_to_fit
    # [ob, a, ob_next, r, done]
    s = np.asanyarray([member[0] for member in remember])
    a = np.asanyarray([member[1] for member in remember])
    s_next = np.asanyarray([member[2] for member in remember])
    r = np.asanyarray([member[3] for member in remember])

    old_q_vals = Q.predict(s, batch_size=32)
    new_q_vals = Q.predict(s_next, batch_size=32)
    
    old_q_vals[range(len(a)), a] = r + gamma * np.amax(new_q_vals, axis=1)
    
    old_q_vals_to_fit = np.concatenate((old_q_vals_to_fit, old_q_vals))
    s_to_fit = np.concatenate((s_to_fit, s))

    if experiments_left == 0:
        experiments_left = 30
        Q.fit(x=s_to_fit, y=old_q_vals_to_fit, epochs=experiments_left, verbose=1)
        s_to_fit = np.zeros(shape=(0,3))
        old_q_vals_to_fit = np.zeros(shape=(0,2))
    else:
        experiments_left -= 1


def act(Q : Model, ob):
    if np.random.random() < epsilon:
        # action_space.sample() is a convenient function to get a random action
        # that is compatible with this given action space.
        a = env.action_space.sample()
        if a < 0:
            return 0
        else:
            return 1
        # print(a)
        return a
    
    prediction = Q.predict(ob.reshape(1,-1))
    a = np.argmax(prediction)
    return a

gamma = 0.99  # Discounting factor

env = gym.make("Pendulum-v0")

observation_size = env.observation_space.shape[0]
n = 32

use_existing_model = True
if use_existing_model:
    # load json and create model
    json_file = open('q.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    q = model_from_json(loaded_model_json)
    # load weights into new model
    q.load_weights("q.h5")
    print("Loaded model from disk")
else:
    q = dense_nn(observation_size, [n,n,2], "")

q.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.mean_squared_error, metrics=['accuracy'])

ob = env.reset()
rewards = []
reward = 0.0

n_steps = 1000000
epsilon = 1.0  # intial chance to apply a random action

remember = []

is_rendering = False
t = 0
global experiments_left
global old_q_vals_to_fit
global s_to_fit
old_q_vals_to_fit = np.zeros(shape=(0,2))
s_to_fit = np.zeros(shape=(0,3))
experiments_left = 0
t_history = []
timing_start = time.time()
for step in range(n_steps):
    a = act(q, ob)
    ob_next, r, done, _ = env.step(env.action_space.low if a == 0 else env.action_space.high)
    # print(r)
    remember.append([ob, a, ob_next, r, done])
    reward += r
    # if r > -0.005 or is_rendering:
    env.render()
    #     is_rendering = True
    if done:
        is_rendering = False
        update_Q(q, remember)
        remember = []
        print(f" {step/float(n_steps) * 100:3.0f} % completed: Episode finished after {t:4d} timesteps with epsilon {epsilon:1.3f} and reward {reward/t:.4f}")
        t_history.append(t)
        t = 0
        rewards.append(reward)
        reward = 0.0
        ob = env.reset()
        if epsilon > 0.01:
            epsilon *= 0.98
    else:
        t += 1
        ob = ob_next

env.close()

# serialize model to JSON
model_json = q.to_json()
with open("q.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
q.save_weights("q.h5")
print("Saved model to disk")

print("finished after ", time.time() - timing_start)
plt.scatter(range(len(rewards)), rewards, marker="x", alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("cumulated reward")
plt.show()