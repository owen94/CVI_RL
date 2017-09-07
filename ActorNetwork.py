'''
This file will implement the Actor Network and CVI
'''

import tensorflow as tf
import numpy as np
import tflearn

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, METHOD,
                 learning_rate, tau, noise, beta, initial_epsilon = 1e-8):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.beta = beta
        self.noise = noise
        # Actor Network
        self.scope1 = 'actor'
        self.scope2 = 'target_actor'
        self.inputs, self.out, self.scaled_out = self.creat_actor_network_cvi(scope=self.scope1)
        #self.network_params = tf.trainable_variables()
        self.mean_params = tf.get_collection(self.scope1)
        self.variance_params = tf.get_collection(self.scope1+'_sigma')

        #TODO: assert method name, e.g., {"SGD", "ADAM", "NOISE+SGD", "NOISE+CVI", "NOISE+ADAM", ...}
        self.method = METHOD
        
        
        assert len(self.mean_params) == len(self.variance_params)
        assert len(self.mean_params) == 6

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = \
            self.creat_actor_network_cvi(scope=self.scope2, target= True)
        self.target_network_params = tf.get_collection(self.scope2)

        assert len(self.target_network_params) == len(self.mean_params)
        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.mean_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        self.sess.run(tf.global_variables_initializer())

        # Combine the gradients w.r.t the mean parameters first, the
        # gradients of the variance can be found from it
        # this mean_gradient is already the minus gradient, for gradient ascent purpose

        self.mean_grads = tf.gradients(
            self.scaled_out, self.mean_params, -self.action_gradient)

        self.gcc = None
        self.initial_epsilon = initial_epsilon

        # then set appropriate optimizers for each method
        #TODO: make learning rates more unified, i.e., self.learning_rate vs 0.01 and 0.0001 below
        if "NOISE" in self.method:
            print(self.method)
            self.variance_grads = tf.gradients(self.scaled_out, self.variance_params, -self.action_gradient)
            
            if self.method == "NOISE+CVI":
                self.optimize_variance = tf.train.GradientDescentOptimizer(learning_rate= self.beta).\
                    apply_gradients(zip(self.variance_grads, self.variance_params))
                self.optimize = self.cvi_update_mean(grad=self.mean_grads,var=self.mean_params, sigma=self.variance_params)
                
            if self.method == "NOISE+CVI_diag":  #probably not use ?, keep it for now
                self.optimize_variance = self.cvi_update_sigma(grad=self.mean_grads,sigma = self.variance_params)
                self.optimize = self.cvi_update_mean(grad=self.mean_grads,var=self.mean_params, sigma=self.variance_params)
                
            elif self.method == "NOISE+SGD":
                self.optimize_variance = tf.train.GradientDescentOptimizer(learning_rate= self.beta).\
                    apply_gradients(zip(self.variance_grads, self.variance_params))
                self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
                    apply_gradients(zip(self.mean_grads, self.mean_params))
                    
            elif self.method == "NOISE+ADAM":       
                self.optimize_variance = tf.train.AdamOptimizer(learning_rate= self.beta).\
                    apply_gradients(zip(self.variance_grads, self.variance_params))
                self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
                    apply_gradients(zip(self.mean_grads, self.mean_params))
        else:
            if METHOD == "SGD":
                self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).\
                    apply_gradients(zip(self.mean_grads, self.mean_params))
            elif METHOD == "ADAM":
                self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
                    apply_gradients(zip(self.mean_grads, self.mean_params))
         
         
        #self.optimize_variance = tf.train.AdagradOptimizer(learning_rate= 0.01, initial_accumulator_value=1e-8).\
        #    apply_gradients(zip(self.variance_grads, self.variance_params))
              
        #self.optimize_variance = self.cvi_update_sigma(grad=self.mean_grads,sigma = self.variance_params)
        #self.optimize = self.cvi_update_mean(grad= self.mean_grads,var=self.mean_params, sigma= self.variance_params)

        # turn off the variance for sanity check.
        # self.variance_grads = tf.gradients(self.scaled_out, self.variance_params, -self.action_gradient)
        #
        # self.optimize_variance = tf.train.AdagradOptimizer(learning_rate= 0.01, initial_accumulator_value=1e-8).\
        #     apply_gradients(zip(self.variance_grads, self.variance_params))
        # self.optimize = tf.train.GradientDescentOptimizer(learning_rate=0.0001).\
        #      apply_gradients(zip(self.mean_grads, self.mean_params))

        ####################################
        ##############CVI Update ###########
        ####################################
        #self.variance_grads = [ - tf.square(self.mean_grads[i]) for i in range(len(self.mean_grads))]
        #self.variance_grads = self.mean_grads
        # self.optimize_variance = tf.train.GradientDescentOptimizer(self.learning_rate).\
        #     apply_gradients(zip(self.variance_grads, self.variance_params))
        #
        # # collection contains all the variance parameters
        # # since we take the reciprocal of a here, we need to add a small value to a since a can be zero sometime.
        #
        # inverse_var_params = [1/a for a in self.variance_params]
        # self.natural_grads = [tf.multiply(a, b) for a, b in zip(inverse_var_params,self.mean_grads)]
        # self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).\
        #     apply_gradients(zip(self.natural_grads, self.mean_params))
        self.num_trainable_vars = len(self.mean_params) + len(self.variance_params) + \
                                  len(self.target_network_params)


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


        state_inputs = tf.placeholder(dtype=tf.float32,shape=(None, self.s_dim),name='input')

        # the truncated initializer is the one used in fully_connected network
        w_initializer = tf.random_normal_initializer(mean=0.,stddev=0.3)
        #w_initializer = tflearn.initializations.truncated_normal()
        sigma_initializer = tf.constant_initializer(value=self.noise)
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

                    # noisy_w = w
                    # noisy_b = b
                    # assert noisy_w == w
                    # assert noisy_b == b
                    if output_layer:
                        layer_output = tf.nn.tanh(tf.matmul(input, noisy_w) + noisy_b)
                    else:
                        layer_output = tf.nn.relu(tf.matmul(input, noisy_w) + noisy_b)
                else:
                    if output_layer:
                        layer_output = tf.nn.tanh(tf.matmul(input, w) + b)
                    else:
                        layer_output = tf.nn.relu(tf.matmul(input, w) + b)

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

    def cvi_update_sigma(self,grad, sigma):

        assert len(grad) == len(sigma)

        variance_grads = [tf.square(grad[i]) for i in range(len(grad))]

        # beta is lambda_1
        update_sigma = [sigma[i].assign(sigma[i] + self.beta * variance_grads[i] ) for i in range(len(sigma))]

        return update_sigma

    def cvi_update_mean(self, grad, var, sigma):

        a = self.sess.run(sigma[0])

        assert np.min(a) >= self.noise

        assert len(grad) == len(var)

        inverse_var_params = [1/a for a in sigma]

        natural_grads = [tf.multiply(a, b) for a, b in zip(inverse_var_params, grad)]

        # learning rate * beta = lambda_2
        update_mean = [var[i].assign(var[i] - self.learning_rate *
                                     natural_grads[i]) for i in range(len(var))]
        return update_mean
        
    def train(self, inputs, a_gradient):
    
        if "NOISE" in self.method:
            self.sess.run(self.optimize_variance, feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            })
            #a = self.sess.run(self.mean_params[0])

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

        # b = self.sess.run(self.mean_params[0])
        #
        # c = self.sess.run(self.check_grad[0])
        #
        # print(a[:,:10])
        # print(b[:,:10])
        # print(c[:,:10])
        # print(a - b - self.learning_rate*c)
        #
        # print('check')


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


    # probably not used ? move to here for now.
    def adagrad(self,grad, var, gcc = None):

        assert len(grad) == len(var)

        if gcc is None:
            self.gcc = [tf.square(grad[i]) for i in range(len(grad))]
        else:
            add_gcc = [tf.square(grad[i]) for i in range(len(grad))]
            self.gcc = [tf.add(a, b) for a, b in zip(self.gcc, add_gcc)]

        adaptive_grad = [ grad_i / tf.sqrt(tf.add(gcc_i, self.initial_epsilon))
                         for grad_i, gcc_i in zip(grad, self.gcc)]

        update = [var[i].assign(var[i] - self.learning_rate * adaptive_grad[i]) for i in range(len(var))]


        return update, adaptive_grad

        
        
        
        
        
        