import tensorflow as tf
from  tensorflow import keras
from  tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x,y), _ = datasets .mnist .load_data ()
x = tf.convert_to_tensor (x,dtype= tf.float32 )  / 255.
y = tf.convert_to_tensor (y,dtype= tf.int32 )

print(x.shape ,y.shape ,x.dtype ,y.dtype )
print(tf.reduce_min(x),tf.reduce_max (x))
print(tf.reduce_min(y),tf.reduce_max (y))

train_db = tf.data .Dataset .from_tensor_slices((x,y)).batch(128)
train_iter =iter(train_db )
sample = next(train_iter)
print('batch:',sample [0].shape,sample[1].shape)





w1 = tf.random .truncated_normal ([784,256])
b1 = tf.zeros([256])
w2 = tf.random .truncated_normal ([256,2128])
b2 = tf.zeros([128])
w3 = tf.random .truncated_normal ([128,10])
b3 = tf.zeros([10])

lr = 0.01

for  step,(x,y) in enumerate ( train_db ):

    x = tf.reshape (x,[-1,28*28])
    with tf.GradientTape() as tape:
        h1 = x@w1 + b1

         h2 = h1@w2 + b2

        out = h2@w3 + b3

        y_onehot = tf.one_hot (y,depth = 10)

        loss = tf.square (y_onehot - out)

        loss = tf.reduce_mean (loss)

    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3 ])

    w1 = w1 - lr * grads[0]
    b1 = b1 - lr * grads[1]
    w2 = w2 - lr * grads[2]
    b2 = b2 - lr * grads[3]
    w3 = w3 - lr * grads[4]
    b3 = b3 - lr * grads[0]

    if step %100 == 0:
        print(step ,'loss:',float(loss))