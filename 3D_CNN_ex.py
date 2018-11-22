from data_prep_preprocess import *
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

width = 120
height = 120
depth = 15
nLabel = 6
num_chan = 1

training_step = 2000
display_step = 1
learning_rate = 0.00001
batch_size = 4
# Placeholders 
x = tf.placeholder(tf.float32, shape=[None, depth,height,width,num_chan],name = 'x') # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel],name = 'y_true')  # [None, 10]

## Weight Initializationbatch
# Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size

## 3D convolution using tensorflow
'''
tf.nn.conv3d(
    input,
    filter,
    strides,
    padding, 
    data_format='NDHWC',
    dilations=[1, 1, 1, 1, 1],
    name=None
)
input: Shape [batch, in_depth, in_height, in_width, in_channels] (by default)
filter: Shape [filter_depth, filter_height, filter_width, in_channels, out_channels]
strides: The stride of the sliding window for each dimension of input(Must have strides[0] = strides[4] = 1)
padding: A string from: "SAME", "VALID"
data_format: An optional string from: "NDHWC", "NCDHW". 
Defaults to "NDHWC". The data format of the input and 
output data. With the default format "NDHWC", the data
is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
Alternatively, the format could be "NCDHW", the data 
storage order is: [batch, in_channels, in_depth, in_height, in_width].
'''
def conv3d(x, W, data_format):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME',data_format = data_format) # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')

def avg_pool_2x2x2(x):  #d/2,h/2,w/2
  return tf.nn.avg_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')

def avg_pool_1x2x2(x): #d,h/2,w/2 
  return tf.nn.avg_pool3d(x, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='VALID')  

## First Convolutional Layer
W_conv1 = weight_variable([12, 3, 3, 1, 64])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([64])  # bias vector for each output channel. = [32]

x_image = x
print(x_image.get_shape) # (?, 1,15,120,120)


h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1,"NDHWC") + b_conv1)  # (?, 120, 120, 15, 16)  
h_pool1 = avg_pool_2x2x2(h_conv1)  #(?, 128, 128, 20, 32)  
#batch_normalization
h_pool1 = tf.layers.batch_normalization(h_pool1)

## Second Convolutional Layer
W_conv2 = weight_variable([1, 3, 3, 64,64]) 
b_conv2 = bias_variable([64]) 

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2,"NDHWC") + b_conv2)   # (?, 128, 128, 20, 64)  
h_pool2 = avg_pool_2x2x2(h_conv2) # (?, 64, 64, 10, 64)    
#batch_normalization
h_pool2 = tf.layers.batch_normalization(h_pool2)


## Densely Connected Layer (or fully-connected layer)

W_fc1 = weight_variable([172800, 1024])  
b_fc1 = bias_variable([1024]) 

h_pool2_flat = tf.reshape(h_pool2, [-1, 172800])   
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  

## Dropout (to reduce overfitting; useful when training very large neural network)
keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
lr = tf.placeholder(tf.float32,name = 'lr')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # -> output: 1024

## Readout Layer
W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print 'y_conv',y_conv.get_shape   # -> output: 10

y_pred = tf.nn.softmax(y_conv,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))#cross_entropy loss function
optimizer = tf.train.AdamOptimizer(learning_rate) #adam optimizer
train_step = optimizer.minimize(cost) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

loss_store = []
saver = tf.train.Saver()
factor = 1
with tf.Session() as sess:
    sess.run(init)
  
    X_tr_array,label = get_kth_data(batch_size)
    for step in range(1,training_step+1):
        batch_x,batch_y = get_kth_data_next_batch(X_tr_array,batch_size,label)
        #shape of batch_x-> [batch_size, num_chan,depth,height,width] 
        sess.run(train_step,feed_dict = {x:batch_x, y_: batch_y, keep_prob: 0.7,lr:learning_rate*factor}) 
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y_: batch_y,keep_prob: 1.0,lr:learning_rate})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))   
            factor*=0.1
        loss_store.append(loss)

    saver.save(sess, "./act_recog_3D_CNN_model.ckpt")         

plt.plot(range(1,training_step+1),loss_store)  
plt.show()          





