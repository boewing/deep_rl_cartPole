import time 
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from collections import defaultdict

def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s_next, a] for a in actions]) 
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low, high)]
                        #  zip(low.flatten(), high.flatten())]
        self.observation_space = Discrete(n_bins ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)

def act(ob):
    if np.random.random() < epsilon:
        # action_space.sample() is a convenient function to get a random action
        # that is compatible with this given action space.
        return env.action_space.sample()

    # Pick the action with highest q value.
    qvals = {a: Q[ob, a] for a in actions}
    max_q = max(qvals.values())
    # In case multiple actions have the same maximum q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param

env = gym.make("CartPole-v1")
actions = range(env.action_space.n)

env = DiscretizedObservationWrapper(
    env, 
    n_bins=10, 
    low=[-2.4, -2.0, -0.42, -3.5], 
    high=[2.4, 2.0, 0.42, 3.5]
)


ob = env.reset()
rewards = []
reward = 0.0

n_steps = 10000000
epsilon = 0.1  # 10% chances to apply a random action


t = 0
t_history = []
timing_start = time.time()
for step in range(n_steps):
    a = act(ob)
    ob_next, r, done, _ = env.step(a)
    update_Q(ob, r, a, ob_next, done)
    reward += r
    if step == 0 or step > n_steps - 1000:
        env.render()
    if done:
        print("Episode finished after {:4d} timesteps".format(t))
        t_history.append(t)
        # print(t)
        t = 0
        rewards.append(reward)
        reward = 0.0
        ob = env.reset()
    else:
        t += 1
        ob = ob_next

print("finished after ", time.time() - timing_start)
plt.scatter(range(len(t_history)), t_history, marker="x", alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("Achieved timesteps")
plt.show()