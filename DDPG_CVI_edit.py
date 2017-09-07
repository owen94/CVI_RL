'''
This file will implement the DDPG and inject parameter space noise.
'''

#to suppress tensorflow system output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# argparse part 
import argparse

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from Replay_buffer import Replay_buffer
import tflearn
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# ===========================
#   Actor and Critic DNNs
# ===========================

from ActorNetwork import ActorNetwork

from CriticNetwork import CriticNetwork

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

def main(_):

    #actor_lr_list = [100]  # learning rate
    #actor_noise = [100000]  # this is S_0, 1/(sigma^2) 0 - 1000000
    #beta_list = [0.1] # beta between 0 to 1
    
    ## parser for the above 3 parameters
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("-a_lr", "--actor_lr", dest="actor_lr", default=0.1, type=float, help="actor parameters learning rate (policy)")
    parser.add_argument("-a_noise", "--actor_noise", dest="actor_noise", default=100000, type=float, help="Actor parameter noise 1/(sigma^2), value in 0 - 1,000,000 ?")
    parser.add_argument("-a_beta", "--actor_beta", dest="actor_beta", default=0.1, type=float, help="Actor parameter noise step size, value in 0 - 1")
    
    ## some parser to make things more convenient in the future
    parser.add_argument("-method", "--method", dest="method_idx", default=0, type=int, help="Index to learning method, default is 0 (NOISE+CVI)")
    parser.add_argument("-c_lr", "--critic_lr", dest="critic_lr", default=0.001, type=float, help="learning rate (critic)")
    parser.add_argument("-gamma", "--gamma", dest="gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("-tau", "--tau", dest="tau", default=0.001, type=float, help="Soft target update param")
    
    parser.add_argument("-me", "--max_episode", dest="max_episode", default=10000, type=int, help="Maximum episodes")
    parser.add_argument("-ms", "--max_step", dest="max_step", default=1000, type=int, help="Maximum steps in an episode")
    parser.add_argument("-s", "--seed", dest="seed", default=0, type=int, help="Random seed")
    parser.add_argument("-e", "--env", dest="env_idx", default=0, type=int, help="Index to environment, if e > 0, it needs mujoco!")
    
    ## name of each method
    method_dict = {-2 : "ADAM", 
                   -1 : "SGD", 
                    0 : "NOISE+CVI",
                    1 : "NOISE+CVI_diag",
                    2 : "NOISE+SGD",
                    3 : "NOISE+ADAM",
    }
    
    ## name of each environment
    env_dict = {-2 : "BipedalWalker-v2",
                -1 : "LunarLanderContinuous-v2", 
                0 : "Pendulum-v0",
                1 : "InvertedPendulum-v1",
                2 : "HalfCheetah-v1",
                3 : "Reacher-v1",
                4 : "Swimmer-v1",
                5 : "Ant-v1",
    }
    
    args = parser.parse_args()
    
    ACTOR_LEARNING_RATE = args.actor_lr
    NOISE = args.actor_noise
    beta = args.actor_beta
    CRITIC_LEARNING_RATE = args.critic_lr
    GAMMA = args.gamma
    TAU = args.tau
    
    MAX_EPISODES = args.max_episode      
    MAX_EP_STEPS = args.max_step   
    RANDOM_SEED = args.seed   
    METHOD = method_dict[args.method_idx]
    ENV_NAME = env_dict[args.env_idx]
            
    # Render gym env during training
    RENDER_ENV = True
    # Use Gym Monitor
    GYM_MONITOR_EN = True
    # Directory for storing gym results
    Result_Path = './result/' + (ENV_NAME) + '/' + METHOD  + '/noise_' + str(NOISE) + '/lr_' + str(ACTOR_LEARNING_RATE) \
                  + '/beta_' + str(beta)

    if not os.path.exists(Result_Path):
        os.makedirs(Result_Path)

    MONITOR_DIR = Result_Path + '/tf_%s' % ENV_NAME
    # Directory for storing tensorboard summary results
    SUMMARY_DIR = Result_Path + '/tf_%s' % ENV_NAME
            
    # Size of replay buffer
    BUFFER_SIZE = 1000000
    MINIBATCH_SIZE = 64


    # content of train() is move to below to avoid passing many input argument from argparse    
        
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

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, METHOD, 
                             ACTOR_LEARNING_RATE, TAU, NOISE, beta)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        # if GYM_MONITOR_EN:
        #     if not RENDER_ENV:
        #         env = wrappers.Monitor(
        #             env, MONITOR_DIR, video_callable=False, force=True)
        #     else:
        #         env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        #train(sess, env, actor, critic, MAX_EPISODES, MAX_EP_STEPS, GAMMA, TAU)

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

                # TODO: Move this part to actor network so that no noise method add action noise when predict.
                # Added exploration noise
                #a = actor.predict(s[np.newaxis,:]) + (1. / (1. + i))
                a = actor.predict(s[np.newaxis,:])
                
                #s2, r, terminal, info = env.step(np.clip(a[0], -1.0, 1.0))
                [s2, r, terminal, info] = env.step(a[0])

                # No future for final time step
                if j == MAX_EP_STEPS - 1 :
                    terminal = 1
                    
                replay_buffer.store_transition(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

                # Keep adding experience to the memory until there are at least minibatch size samples
                if replay_buffer.size() > MINIBATCH_SIZE:
                    [s_batch, a_batch, r_batch, t_batch, s2_batch] = replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # Calculate targets based on the action from the target actor
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    #this may be faster since it does not explicitly use loop
                    # d_idx = np.where(1 == t_batch)
                    # target_q[d_idx] = 0
                    # y_i = r_batch + GAMMA*target_q
                    #
                    y_i = []
                    for k in range(MINIBATCH_SIZE):
                       if t_batch[k]:
                           y_i.append(r_batch[k])
                       else:
                           y_i.append(r_batch[k] + GAMMA * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

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
            #
            # if i % 100 ==0:
            #     plt.plot(episode_reward)
            #     plt.xlabel('Episode')
            #     plt.ylabel('Rewards')
            #     savepath = './cvi/lunar/cvi_n_' + \
            #                str(actor.noise) + '_lr_' + str(actor.learning_rate) + '_b_' + str(actor.beta) +  '.png'

                #savepath = './cvi/lunar/sgd_0.0001_noise_10000.png'
                # plt.savefig(savepath)
                # plt.pause(0.01)

                path = Result_Path +   '/reward.npy'
                #result_path = './cvi/lunar/sgd_0.0001_noise_10000.npy'
                np.save(path, episode_reward)

        #
        # if GYM_MONITOR_EN:
        #     env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
