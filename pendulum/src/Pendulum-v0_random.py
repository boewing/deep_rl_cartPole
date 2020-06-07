import gym
env = gym.make('Pendulum-v0')
n_steps = 1000
step = 0
while True:
    observation = env.reset()
    for t in range(200):
        step += 1
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print(f" {step/float(n_steps) * 100:3.0f} % completed: Episode finished after {t:4d} timesteps")
            break
    
    if step > n_steps:
        break
env.close()