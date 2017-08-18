import numpy as np
import tensorflow as tf
import pandas as pd
import random
import gym
import matplotlib.pyplot as plt
from maze_env import  Maze

np.random.seed(1)
tf.set_random_seed(1)

class DQN(object):
    def __init__(self, state_dim,
                 n_actions,
                 memory_size=1000000,
                 batch_size=64,
                 reward_decay=0.99,
                 replace_target_steps=1000,
                 learning_rate = 0.001,
                 epsilon=0.9,
                 show_tensorboard=False,
                 e_greedy_increment=None
                 ):

        self.state_dim = state_dim # input dimension
        self.n_actions = n_actions

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.reward_decay = reward_decay
        self.replace_target_steps = replace_target_steps
        self.learning_rate=learning_rate
        self.epsilon_max = epsilon
        self.show_tensorboard = show_tensorboard
        self.epsilon_increment = e_greedy_increment

        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.cost_his = []
        self.learn_step = 0 # the number of learn_steps which can help monitor update Q-network

        self.memory = np.zeros(shape=(self.memory_size, 2*self.state_dim+2))
        self.done = False


        # Initialize the Q-network
        self.build_dqn()

        # Initialize the Q-network parameters
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.show_tensorboard:   # show the computational graph
            tf.summary.FileWriter("logs/", self.sess.graph)


    def build_dqn(self):

        def build_network(state, collection_name, n_neurons, w_initializer, b_initializer):
            with tf.variable_scope('layer_1'):

                w1 = tf.get_variable(name='w1',shape=(self.state_dim,n_neurons),dtype=tf.float32,
                                     initializer=w_initializer,collections=collection_name)
                b1 = tf.get_variable(name='b1',shape=(1, n_neurons),dtype=tf.float32,
                                     initializer= b_initializer, collections=collection_name)
                layer_1_output = tf.nn.relu(tf.matmul(state, w1) + b1)

            with tf.variable_scope('layer_2'):
                w2 = tf.get_variable(name='w2',shape=(n_neurons,self.n_actions),dtype=tf.float32,
                                     initializer=w_initializer,collections=collection_name)
                b2 = tf.get_variable(name='b2',shape=(1, self.n_actions),dtype=tf.float32,
                                     initializer= b_initializer, collections=collection_name)
                output = tf.matmul(layer_1_output, w2) + b2

                return output

        with tf.name_scope('input'):
            self.eval_s = tf.placeholder(dtype=tf.float32,shape=(None,self.state_dim),name='eval_s')
            self.target_s = tf.placeholder(tf.float32, [None, self.state_dim], name='target_s')
            self.r = tf.placeholder(tf.float32, [None, ], name='r')
            self.a = tf.placeholder(tf.int32, [None, ], name='a')

        n_layer_1 = 100
        w_initializer = tf.random_normal_initializer(mean=0.,stddev=0.3)
        b_initializer = tf.constant_initializer(value=0.1)

        # build the evaluate network
        with tf.variable_scope('eval_net'):

            collection_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.eval_q = build_network(state=self.eval_s, collection_name = collection_name,n_neurons=n_layer_1,
                                        w_initializer=w_initializer,b_initializer=b_initializer)
        # build the target network, it must be the same as the evaluate network
        with tf.variable_scope('target_nets'):
            collection_name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.target_q = build_network(state=self.target_s, collection_name = collection_name,n_neurons=n_layer_1,
                                        w_initializer=w_initializer,b_initializer=b_initializer)

        with tf.variable_scope('max_target_q'):
            if self.done:
                estimate_q = self.r
            else:
                estimate_q = self.r + self.reward_decay * tf.reduce_max(self.target_q, axis=1, name='Qmax_s_')    # shape=(None, )
            self.max_target_q = tf.stop_gradient(input=estimate_q)

        with tf.variable_scope('one_hot_q'):
            a_one_hot = tf.one_hot(self.a, depth=self.n_actions, dtype=tf.float32)
            self.q_eval_wrt_a = tf.reduce_sum(self.eval_q * a_one_hot, axis=1)     # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval_wrt_a,self.max_target_q))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


    def update_dqn(self, done = False):
        # update the target q network
        if self.learn_step % self.replace_target_steps == 0:
            self.update_target_net()
            print('Target network is updated in step {}........'.format(self.learn_step))

        # Get samples from the replay buffer
        if self.memory_counter > self.memory_size:
            sample_index = random.sample(range(self.memory_size),self.batch_size)
        else:
            sample_index = random.sample(range(self.memory_counter), self.batch_size)

        input_samples = self.memory[sample_index,:]

        self.done = done

        # update the Q_network
        _, cost = self.sess.run([self.train_op,self.loss],
                                 feed_dict={
                self.eval_s:     input_samples[:, :self.state_dim],
                self.a:          input_samples[:, self.state_dim],
                self.r:          input_samples[:, self.state_dim + 1],
                self.target_s:   input_samples[:, -self.state_dim:],
            })

        self.learn_step += 1
        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def update_target_net(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        for t, e in zip(t_params, e_params):
            assign = tf.assign(t,e)
            self.sess.run(assign)
        # a general way is self.sess.run([tf.assign(t,e) for t, e in zip(t_params, e_params) ])


    def store_transitions(self, eval_s, a ,r, target_s):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((eval_s, [a, r],target_s))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        # choose action with epsilon greedy strategy
        # use the evaluate network to choose the action, not the target network. Hence, if
        # we want to do parameters space noise, we should inject the noise into this step.
        if np.random.uniform(low=0,high=1) < self.epsilon:
            action_prob = self.sess.run(self.eval_q, feed_dict={self.eval_s:state})
            action = np.argmax(action_prob)
        else:
            action = np.random.randint(low=0,high=self.n_actions)
        return action

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(1)

    sess = tf.Session()
    state_dim   = env.observation_space.shape[0]
    num_actions = env.action_space.n


    DQN = DQN(state_dim= state_dim, n_actions=num_actions,show_tensorboard=True)

    episode_reward = []

    for i_episode in range(10000):

        state = env.reset()
        total_rewards = 0
        steps = 0

        while True:
            env.render()
            action = DQN.choose_action(state=state[np.newaxis,:])
            next_state, reward, done, _ = env.step(action)
            DQN.store_transitions(eval_s= state,a=action,r=reward,target_s=next_state)
            state = next_state

            if DQN.memory_counter > DQN.batch_size:
                DQN.update_dqn(done= done)
            if done:
                print('The game in episode {} finished with {} timesteps.'.format(i_episode, steps ))
                episode_reward += [steps]
                # if steps < 199:
                #     print('The reward is when hit the flag is {}'.format(reward))
                break
            steps += 1

        if i_episode%100 ==0:
            plt.plot(episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Steps for success')
            plt.savefig('Cartpole')
            plt.pause(3)








