'''
play with the OpenAI gym environment now.
'''
import gym
env = gym.make('CartPole-v0')
for i_episode in range(2):
    observation = env.reset()
    for t in range(1):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.step(env.action_space.sample())
            break
