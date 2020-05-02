import gym
env = gym.make('CartPole-v0')
# env = gym.make('Acrobot-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        env.render()
        # print(observation)
        angle = observation[2]
        d_angle = observation[3]
        if angle + d_angle < 0:
            action = 0
        else:
            action = 1
        # action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()