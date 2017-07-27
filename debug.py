
import tensorflow as tf

x = tf.get_variable(name='x',shape=None,initializer=[[1.0, 2.0, 3.0],[1,2,3]])

samples = tf.identity(input=x)
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)


x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x)
init = tf.initialize_all_variables()



with tf.Session() as session:
    init.run()
    for i in range(5):
        print(y.eval())
        print(x, x_plus_1, y)

#print(sess.run(samples))
