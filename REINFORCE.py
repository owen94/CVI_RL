'''
In this file, we will implement the REINFORCE algorithm: Monte-Carlo Policy Gradient with
OpenAI gym and tesnforflow.
'''

import gym
import itertools
import matplotlib
import numpy as np
import sys, random
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
observation = env.reset()
env.render()
print(observation)

class REINFORCE(object):
    def __init__(self, session,
                 policy_network,
                 optimizer,
                 input_dim,
                 num_actions,
                 discount_factor = 0.99,
                 regularization = 0,
                 ):
        self.sess = session
        self.optimizer = optimizer
        self.policy_network = policy_network # policy network take as a tf graph for computation

        # openAI gym environment
        self.input_dim = input_dim
        self.num_actions = num_actions

        # RL algorithm parameters
        self.discount_factor = discount_factor

        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

        self.construct_graph()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))


    def construct_graph(self):
        with tf.name_scope('model_input'):
            self.states = tf.placeholder(dtype=tf.float32,shape=(None,self.input_dim), name = 'states')
            self.sample_action = tf.placeholder(dtype=tf.int32,shape=(None,),name='sample_action')
            self.discount_reward = tf.placeholder(dtype=tf.float32, shape = (None,), name = 'discount_reward')

        self.policy_output = self.policy_network(self.states)
        self.all_act_prob = tf.nn.softmax(self.policy_output, name='act_prob')

        #self.action_scores = tf.identity(input=self.policy_output)
        # compute the loss and gradients, input is the: sample_action(A_t), reward(G_t), and states
        with tf.name_scope('loss_gradients'):
            # compute the cross entropy loss
            entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits\
                (labels=self.sample_action,logits=self.policy_output)  # This cross entropy is equivalent to
            # p(a_t|s_t) in the REINFORCE algorithm.
            self.loss = tf.reduce_mean(entropy_loss * self.discount_reward)

        # train the policy network
        with tf.name_scope('train_network'):
            #self.train_op =  self.optimizer.apply_gradients(self.gradients)
            self.train_op =  self.optimizer.minimize(self.loss)


    def get_action(self,states):
        # Here we first explore the action space with \eplison greedy strategy.
        # use the epsilon-greedy function
        # if random.random() < self.exploration:
        #     return random.randint(0, self.num_actions-1)
        # else:
        action_scores = self.sess.run(self.all_act_prob, {self.states: states})
        action = np.random.choice(range(action_scores.shape[1]), p=action_scores.ravel())
        return action

    def get_update(self):
        # In the function, we update the police network for an episode
        # computing G_t which is the expected future reward that would be used for
        r = 0
        T = len(self.reward_buffer)
        discounted_rewards = np.zeros(T)

        for i in reversed(range(T)):
            r = self.reward_buffer[i] + self.discount_factor * r
            discounted_rewards[i] = r
        # discounted_rewards -= np.mean(discounted_rewards)
        # discounted_rewards /= np.std(discounted_rewards)

        for t in range(T-1):
            states  = self.state_buffer[t][np.newaxis, :] # why need to add a newaxis here??
            actions = np.array([self.action_buffer[t]])
            rewards = np.array([discounted_rewards[t]])

            #grad = [grad for grad, var in self.gradients]

            self.sess.run(self.train_op, {
                self.states: states,
                self.discount_reward: rewards,
                self.sample_action:actions
            })

        self.cleanUp()

    def get_episode_update(self):

        discount_rewards = self.get_discount_reward()
        self.sess.run(self.train_op, {
                self.states: np.vstack(self.state_buffer),
                self.discount_reward: discount_rewards,
                self.sample_action: np.array(self.action_buffer)
            })
        self.cleanUp()


    def get_discount_reward(self):
        r = 0
        T = len(self.reward_buffer)
        discounted_rewards = np.zeros(T)
        for i in reversed(range(T)):
            r = self.reward_buffer[i] + self.discount_factor * r
            discounted_rewards[i] = r
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards

    def cleanUp(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

    def store_episode(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

def train_Games():

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(1)
    #env = env.unwrapped

    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
    state_dim   = env.observation_space.shape[0]
    num_actions = env.action_space.n


    def policy_network(states):
    # define policy neural network
    #       W1 = tf.get_variable("W1", [state_dim, 20],
    #                              initializer=tdamf.random_normal_initializer(mean=0,stddev=0.3))
    #       b1 = tf.get_variable("b1", [20],
    #                              initializer=tf.constant_initializer(0))
    #       h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
    #       W2 = tf.get_variable("W2", [20, num_actions],
    #                              initializer=tf.random_normal_initializer(mean=0,stddev=0.3))
    #       b2 = tf.get_variable("b2", [num_actions],
    #                              initializer=tf.constant_initializer(0))
    #       p = tf.matmul(h1, W2) + b2

        layer = tf.layers.dense(
            inputs=states,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units= num_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        return all_act


    pg_reinforce = REINFORCE(session=sess,optimizer=optimizer, input_dim= state_dim,num_actions=num_actions,
                             policy_network=policy_network)


    episode_reward = []

    for i in range(1000):
        state = env.reset()
        total_rewards = 0
        #episode_length = 0

        for t in range(1000):
            env.render()
            action = pg_reinforce.get_action(states=state[np.newaxis,:])
            next_state, reward, done, _ = env.step(action)
            pg_reinforce.store_episode(state=state, action=action, reward=reward)

            #episode_length += 1

            state = next_state
            total_rewards += reward

            if done:
                if i%20 == 0:
                    print('Episode {} runs for {} timesteps with reward {}.'.format(i, t, total_rewards))
                break


        episode_reward += [total_rewards]
        pg_reinforce.get_episode_update() # in this step, we update the policy for an around.

    plt.plot(episode_reward)
    plt.savefig('reward.png')
    plt.show()


if __name__ == '__main__':

    train_Games()












