
import tensorflow as tf

x = tf.get_variable(name='x',shape=None,initializer=[[1.0, 2.0, 3.0]])

samples = tf.identity(input=x)
init = tf.global_variables_initializer()

y = x + 1
z = y +1  +x

sess = tf.Session()
sess.run(init)
print(sess.run(y))
print(sess.run(z))






#print(sess.run(samples))
