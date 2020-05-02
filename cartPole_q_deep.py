import time 
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from collections import defaultdict
from keras import layers, Model
import keras 

def dense_nn(input_dim, layers_sizes, scope_name):
    """Creates a densely connected multi-layer neural network.
    inputs: the input tensor
    layers_sizes (list<int>): defines the number of units in each layer. The output 
        layer has the size layers_sizes[-1].
    """
    model = keras.Sequential()
    for layer_size in layers_sizes:
        model.add(layers.Dense(layer_size, input_dim=input_dim, activation=keras.activations.relu))
        input_dim = None
    
    return model


def update_Q(Q : Model, remember):
    # [ob, a, ob_next, r, done]
    s = np.asanyarray([member[0] for member in remember])
    a = np.asanyarray([member[1] for member in remember])
    s_next = np.asanyarray([member[2] for member in remember])
    r = np.asanyarray([member[3] for member in remember])

    old_q_vals = Q.predict(s, batch_size=1)
    new_q_vals = Q.predict(s_next, batch_size=1)
    
    old_q_vals[range(len(a)),a] = r + gamma * np.amax(new_q_vals, axis=1)
    hist = Q.fit(x=s, y=old_q_vals, epochs=10, verbose=0)


def act(Q : Model, ob):
    if np.random.random() < epsilon:
        # action_space.sample() is a convenient function to get a random action
        # that is compatible with this given action space.
        a = env.action_space.sample()
        # print(a)
        return a
    
    prediction = Q.predict(ob.reshape(1,-1))
    a = np.argmax(prediction)
    # print(a, prediction)
    return a


gamma = 0.99  # Discounting factor

env = gym.make("CartPole-v0")
actions = range(env.action_space.n)

observation_size = env.observation_space.shape[0]
n = 16
q = dense_nn(observation_size, [n,n,2], "")
q_target = dense_nn(observation_size, [n,n,2], "")

q.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.mean_squared_error, metrics=['accuracy'])

ob = env.reset()
rewards = []
reward = 0.0

n_steps = 1000000
epsilon = 1  # 10% chances to apply a random action

remember = []

t = 0
t_history = []
timing_start = time.time()
for step in range(n_steps):
    a = act(q, ob)
    ob_next, r, done, _ = env.step(a)
    r = 1/(0.1 + abs(ob[2]) + 0.1 *abs(ob[0])) - 6
    # print(r)
    remember.append([ob, a, ob_next, r, done])
    reward += r
    # env.render()
    if done:
        update_Q(q, remember)
        remember = []
        print("Episode finished after {} timesteps".format(t))
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

print("finished after ", time.time() - timing_start)
plt.scatter(range(len(t_history)), t_history, marker="x", alpha=0.2)
plt.show()