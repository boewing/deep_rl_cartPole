import gym
env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {:4d} timesteps".format(t+1))
            break

env.close()