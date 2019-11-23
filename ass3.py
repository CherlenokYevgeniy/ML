import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01

training_epochs =  100
x_train = np.linspace(0,10,100)
y_train = x_train + np.random.normal(0,1,100)

tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

def h(X,w1,w0):
    return tf.add(tf.multiply(X,w1),w0)

w0 = tf.Variable(0.0, name="weights")
w1 = tf.Variable(0.0, name="weights")
y_predicted = h(X,w1,w0)
costF = 0.5*tf.square(Y-y_predicted)

train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(costF)
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for(x,y) in zip(x_train,y_train):
        sess.run(train_op,feed_dict={X:x,Y:y})

w_val_0 = sess.run(w0)
w_val_1 = sess.run(w1)
sess.close()

plt.scatter(x_train,y_train)
y_learned = x_train * w_val_1 + w_val_0
plt.plot(x_train,y_learned,'r')
plt.show()