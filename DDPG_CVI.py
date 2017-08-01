'''
This file will implement the DDPG and inject parameter space noise.
'''

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from Replay_buffer import Replay_buffer
import tflearn
import matplotlib.pyplot as plt
# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 20000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.01

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
#ENV_NAME = 'MountainCarContinuous-v0'
#ENV_NAME = 'LunarLanderContinuous-v2'
ENV_NAME ='Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_pendu'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_pendu'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.scope1 = 'actor'
        self.scope2 = 'target_actor'
        self.inputs, self.out, self.scaled_out = self.creat_actor_network_cvi(scope=self.scope1)
        #self.network_params = tf.trainable_variables()
        self.mean_params = tf.get_collection(self.scope1)
        self.variance_params = tf.get_collection(self.scope1+'_sigma')

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = \
            self.creat_actor_network_cvi(scope=self.scope2, target= True)
        self.target_network_params = tf.get_collection(self.scope2)

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.mean_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients w.r.t the mean parameters first, the
        # gradients of the variance can be found from it
        # this mean_gradient is already the minus gradient, for gradient ascent purpose

        self.mean_grads = tf.gradients(
            self.scaled_out, self.mean_params, -self.action_gradient)

        self.variance_grads = [- tf.square(self.mean_grads[i]) for i in range(len(self.mean_grads))]
        # self.variance_grads = self.mean_grads
        self.optimize_variance = tf.train.GradientDescentOptimizer(self.learning_rate).\
            apply_gradients(zip(self.variance_grads, self.variance_params))

        # collection contains all the variance parameters
        # since we take the reciprocal of a here, we need to add a small value to a since a can be zero sometime.
        inverse_var_params = [1/a for a in self.variance_params]
        self.natural_grads = [tf.multiply(a, b) for a, b in zip(inverse_var_params,self.mean_grads)]

        self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).\
            apply_gradients(zip(self.natural_grads, self.mean_params))

        self.num_trainable_vars = len(self.mean_params) + len(self.variance_params) + len(self.target_network_params)

        self.sess.run(tf.global_variables_initializer())

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


    def creat_actor_network_cvi(self, scope, target = False):

        state_inputs = tf.placeholder(dtype=tf.float32,shape=(None,self.s_dim),name='input')

        w_initializer = tf.random_normal_initializer(mean=0.,stddev=0.3)
        sigma_initializer = tf.constant_initializer(value=0.01)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        b_initializer = tf.constant_initializer(value=0.1)
        n_layer1 = 400
        n_layer2 = 300
        collection = [scope, tf.GraphKeys.GLOBAL_VARIABLES]

        def build_layer(layer_scope,dim_1,dim_2,input,collections, output_layer, w_initializer = w_initializer):
            with tf.variable_scope(layer_scope):
                if output_layer:
                    w_initializer = w_init
                w = tf.get_variable(name='w',shape=(dim_1,dim_2),dtype=tf.float32,
                                 initializer=w_initializer,collections=collections)
                b = tf.get_variable(name='b',shape=(1, dim_2),dtype=tf.float32,
                                 initializer= b_initializer, collections=collections)
                if not target:
                    eps_w = tf.random_normal(shape=(dim_1,dim_2), mean=0, stddev=1, dtype=tf.float32)
                    sigma_w = tf.get_variable(name='sigma_w',shape=(dim_1,dim_2),dtype=tf.float32,
                                 initializer=sigma_initializer,collections=[scope+'_sigma', tf.GraphKeys.GLOBAL_VARIABLES])
                    noisy_w = w + eps_w * (1/tf.sqrt(sigma_w))

                    eps_b = tf.random_normal(shape=(1,dim_2), mean=0, stddev=1,dtype=tf.float32)
                    sigma_b = tf.get_variable(name='sigma_b',shape=(1,dim_2),dtype=tf.float32,
                                 initializer=sigma_initializer,collections=[scope+'_sigma', tf.GraphKeys.GLOBAL_VARIABLES])
                    noisy_b = b + eps_b * (1/tf.sqrt(sigma_b))
                    if output_layer:
                        layer_output = tf.nn.tanh(tf.matmul(input, noisy_w) + noisy_b)
                    else:
                        layer_output = tf.nn.relu(tf.matmul(input, noisy_w) + noisy_b)
                else:
                    if output_layer:
                        layer_output = tf.nn.relu(tf.matmul(input, w) + b)
                    else:
                        layer_output = tf.nn.tanh(tf.matmul(input, w) + b)

            return layer_output

        with tf.variable_scope(scope):

            layer_1_output = build_layer(layer_scope='layer_1',dim_1=self.s_dim,dim_2=n_layer1,input=state_inputs,
                                         collections=collection,output_layer=False)
            layer_2_output = build_layer(layer_scope='layer_2',dim_1=n_layer1,dim_2=n_layer2,input=layer_1_output,
                                         collections=collection,output_layer=False)
            output = build_layer(layer_scope='layer_3',dim_1=n_layer2,dim_2=self.a_dim,input=layer_2_output,
                                         collections=collection,output_layer=True)

            scaled_out = tf.multiply(output, self.action_bound)

        return state_inputs, output, scaled_out


    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize_variance, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================


def train(sess, env, actor, critic):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = Replay_buffer(BUFFER_SIZE, RANDOM_SEED)
    episode_reward = []
    episode_q = []

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        ep_ave_q = []

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            # Added exploration noise
            a = actor.predict(s[np.newaxis,:])

            #s2, r, terminal, info = env.step(np.clip(a[0], -1.0, 1.0))
            s2, r, terminal, info = env.step(a[0])

            # if terminal:
            #     r += 100

            replay_buffer.store_transition(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets based on the action from the target actor
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)
                ep_ave_q += [np.mean(predicted_q_value)]

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)

                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('Episode {} end with {} steps. Cummulative reward is {}, final step reward is {}. Average episode '
                      'q_value is {}, cumulative max q_value is {}'
                      .format(i, j, int(ep_reward), r, np.mean(ep_ave_q), ep_ave_max_q))

                break
        episode_reward += [ep_reward]
        episode_q += [np.mean(ep_ave_q)]

        if i%50 ==0:
            print('Average reward in the first {} episodes is {}.'.format(i, np.mean(np.array(episode_reward))))

        if i % 100 ==0:
            plt.plot(episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Rewards')
            plt.savefig('pendu_100')
            plt.pause(0.01)

        if i % 100 ==0:
            plt.plot(episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Q-value')
            plt.savefig('pendu_q_100')
            plt.pause(0.01)

def main(_):
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        #assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
