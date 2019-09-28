# conda activate py36gputf_b && \
# cd /home/young && \
# rm e.l && python temp.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Install matplotlib to use matplotlib's graph visualization by running
# pip install matplotlib
import matplotlib.pyplot as plt

# ================================================================================
tf.reset_default_graph()

# ================================================================================
mnist=load_digits()
# print("mnist",mnist)

# ================================================================================
X,y=mnist.data,mnist.target

# ================================================================================
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
# print("X_train",X_train.shape)
# print("y_train",y_train.shape)
# print("X_test",X_test.shape)
# print("y_test",y_test.shape)
# (1437, 64)
# (1437,)
# (360, 64)
# (360,)

# (1437, 64) means 1473 number of train images, each image is transformed to (64,) shape 1D array from (8,8) 2D array

# ================================================================================
n_labels=len(np.unique(y))
# print("n_labels",n_labels)
# 10

# Following image shows example of MNIST dataset
# https://www.google.com/search?q=mnist+dataset&sxsrf=ACYBGNSqEo4W9dpAVDos7LT1W2g_zTUyUg:1569709681524&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjlx96fyPTkAhWrzIsBHZS_ANEQ_AUIEigB&biw=1911&bih=687#imgrc=83-3_IxDBeLMIM:

# Your goal is to predict "number's label" by training neural network using pairs of drawn_number-label dataset

# ================================================================================
batch_size=32

n_batch_train=len(X_train)//batch_size
n_batch_test=len(X_test)//batch_size
# print("n_batch_train",n_batch_train)
# print("n_batch_test",n_batch_test)
# 44
# 11

# ================================================================================
# Placeholder for input data
# None means batch size (if you use 10 batch, None is assigned by 10 dynamically, if you use 20, None is assigned by 20)
# 64 means input data's shape. MNIST image is 8 pixels * 8 pixels image.
# 64 shape 1D array is created from 8*8 image by flatting 8*8 image (2D array)
# float32 means data type in input data array is float32
X_input=tf.placeholder("float32",(None,64))

# This is a placeholder also for input data
# But this input data is "label" than "image"
# "label" means target number like 1 (when MNIST image shows "number 1"), 2 (when MNIST image shows "number 2"), ...
y_input=tf.placeholder("int32",(None))

# ================================================================================
# You use one fully connected layer
# That fully connected layer takes in X_input as input data
# That fully connected layer outputs (128,) shape 1D array
# After you get output from fully connected layer, that output is passed to ReLU activation function to give non-liearity characteristics to your output
z2=tf.layers.dense(X_input,128,activation=tf.nn.relu)

# You can try to add more layers (in this case, you will add more fully connected layers after above fully connected layer)
# Below fully connected layer takes in z2 (output from above fully connected layer)
# Below fully connected layer outputs (64,) predicted 1D array output result
z3=tf.layers.dense(z2,64,activation=tf.nn.relu) # Added fully connected layer

z4=tf.layers.dense(z3,32,activation=tf.nn.relu) # Added fully connected layer

# This is also fully connected layers but it's last fully connected layer
# last fully connected layer has "shape of target array"
# For example, n_labels=10 in this example (because MNIST image data is conposed of 10 number digits)
logits=tf.layers.dense(z4,n_labels)

# You can use sigmoid, softmax, etc
# In this case, the code uses softmax layer
# softmax layer outputs probability values where summation of all probability values should be 1
# After softmax, you can find array like [0.01,0.9,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
# Second element has highest probability value (0.9)
# And you have label array like [0,1,2,3,4,5,6,7,8,9]
# Since you have highest probability value in second index (0.9 one), you will index second index element from [0,1,2,3,4,5,6,7,8,9]
# So, neural network prediction is "number 1"
prediction=tf.nn.softmax(logits)

# ================================================================================
# Now, you will calculate loss value
# loss value = prediction - actual_label
# high loss: your neural network is not predicting accurately

# So, you should adjust the parameters of your neural network towards the direction where loss_value decreases

# When you adjust the parameters of your neural network (which is called "updating neural network"), you use optimizer like SGD, ADAM etc

# Updating processing using optimizer uses "partial derivative", "back propagation", "learning rate", "chain rule", etc which is managed by deep learning frameworks like TensorFlow, PyTorch, Keras, Caffe, etc

# ================================================================================
# Use cross entropy loss function (there are various loss function but cross entropy is much used for classification problem)

loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=tf.one_hot(y_input,depth=n_labels))

# ================================================================================
# Prepare ADAM optimizer and call minimize() to minimize "loss" 

train_op=tf.train.AdamOptimizer().minimize(loss)

# ================================================================================
# Following code is TensorFlow's unique characteristics (static structure of neural network)

# In above steps, you built neural network

# And you will execute built-network in following code

# PyTorch doesn't require this step, because PyTorch can execute "neural network" code as you are write the code just like you create other application like mobile app, web app, desktop app
# So, PyTorch is easy to write neural network code and easy to test it

# ================================================================================
# with tf.Session() as sess:

#   # You initialize global variables (I guess global variables are X_input, y_input, z2, z3, ...)
#   sess.run(tf.global_variables_initializer())

#   # You perform 10 epochs training
#   for epoch in range(10):
#     y_pred_array=[]

#     for b in range(n_batch_train):
#       # You pass real data into placeholders like X_input, y_input
#       # By using Python indexing, you create batchsized input data, and you pass indexed data into placeholder

#       # sess.run([train_op,loss] means you execute built-graph-neural-network up to "train_op" and "loss" nodes and you get the return value from those nodes
#       _,loss_=sess.run([train_op,loss],feed_dict={X_input:X_train[b*batch_size:(b+1)*batch_size],
#                                                   y_input:y_train[b*batch_size:(b+1)*batch_size]})

#     # Sometimes (if epoch%2==0), you print the loss values
#     if epoch%2==0:
#       print("loss_",loss_.shape)
#       # loss_ (32,)

#       # Following 1D array indicates 32 number of loss values with respect to 32 number of predictions from NN

#       print("loss_",loss_)
      # loss_ [1.40508460e-02 5.60784042e-02 5.44182348e+00 3.66608202e-02
      # 1.10286683e-01 2.04012588e-01 1.62038505e-02 1.14068007e-02
      # 2.90275842e-01 3.72954272e-02 1.68496042e-01 3.07307214e-01
      # 2.47403886e-02 1.50750875e-02 1.44054279e-01 3.28463125e+00
      # 8.71928930e-02 8.55136395e-01 1.75087705e-01 1.68514669e-01
      # 1.52941803e-02 8.64489644e-04 1.78042091e-02 4.22656164e-03
      # 2.84921145e-03 4.02213275e-01 3.27670686e-02 7.27089960e-03
      # 1.10337384e-01 2.33557150e-01 8.38549256e-01 3.51966715e+00]
      # loss_ [8.6360313e-03 2.3069095e-02 3.2940047e+00 1.3642537e-03 9.9864379e-03
      # 5.7214085e-02 1.6481400e-03 6.0686134e-03 1.4130755e-01 2.4066791e-03
      # 5.2415803e-03 2.7701668e-02 1.6104121e-03 2.7702788e-03 8.9158215e-02
      # 1.2692045e+00 8.8615017e-03 2.2857369e-01 3.8795097e-04 6.4575342e-03
      # 1.0583758e-03 1.9381552e-04 8.1851351e-04 1.5767297e-03 1.8542492e-03
      # 1.5473422e-01 1.7885009e-02 1.0320581e-03 2.2359999e-02 1.1656626e-02
      # 6.5447651e-02 2.3541777e+00]
      # loss_ [7.4314815e-03 3.3547446e-02 1.2528081e+00 1.6675988e-04 2.7362783e-03
      # 2.4146289e-02 6.7318667e-04 2.1614302e-03 1.2092289e-01 1.4934113e-03
      # 8.8366552e-04 6.8911742e-03 7.7527505e-04 4.6528480e-04 5.7329021e-02
      # 7.0989662e-01 2.9087635e-03 7.2695561e-02 4.9828242e-05 7.4537622e-04
      # 4.7803417e-04 6.9377398e-05 3.2979771e-04 7.0404058e-04 6.7866658e-04
      # 3.7066057e-02 1.8042723e-02 5.4201693e-04 1.5150942e-02 5.6765778e-03
      # 2.0159472e-02 8.1129718e-01]
      # loss_ [3.7515040e-03 3.3839744e-02 3.7100658e-01 7.3907031e-05 1.1842389e-03
      # 7.1568079e-03 3.7365133e-04 2.2279222e-03 6.6854544e-02 1.3048477e-03
      # 3.3730539e-04 3.8036394e-03 4.7672351e-04 2.4268066e-04 1.8586401e-02
      # 5.1187027e-01 1.0319391e-03 3.0258046e-02 1.5020258e-05 1.5126515e-04
      # 2.6294112e-04 2.7536968e-05 1.5674793e-04 1.1344671e-03 6.1004621e-04
      # 7.1744430e-03 1.7052224e-02 3.1871482e-04 7.4189389e-03 3.8878345e-03
      # 5.0955904e-03 3.1749907e-01]
      # loss_ [1.9502683e-03 3.4427539e-02 1.2908670e-01 3.7192607e-05 6.3041836e-04
      # 2.0283142e-03 1.9274284e-04 1.9100533e-03 3.9866488e-02 1.1906686e-03
      # 1.9834458e-04 2.1826036e-03 2.3517227e-04 1.6628313e-04 8.0324011e-03
      # 2.5798002e-01 5.0627289e-04 1.6902557e-02 5.8412379e-06 4.2079995e-05
      # 1.4435203e-04 1.2278481e-05 7.6052631e-05 1.4654384e-03 4.4633917e-04
      # 2.7954807e-03 8.3147548e-03 1.9977480e-04 6.6669174e-03 2.3926462e-03
      # 1.5661367e-03 2.0317224e-01]


# ================================================================================
loss_values_list=[]

# I added this code to see "loss decreasing"
# You should above code from "with tf.Session() as sess:" to "print("loss_",loss_)" to use following code
with tf.Session() as sess:

  # You initialize global variables (I guess global variables are X_input, y_input, z2, z3, ...)
  sess.run(tf.global_variables_initializer())

  # You perform 10 epochs training
  for epoch in range(1000):
    y_pred_array=[]

    for b in range(n_batch_train):
      # You pass real data into placeholders like X_input, y_input
      # By using Python indexing, you create batchsized input data, and you pass indexed data into placeholder

      # sess.run([train_op,loss] means you execute built-graph-neural-network up to "train_op" and "loss" nodes and you get the return value from those nodes
      _,loss_=sess.run([train_op,loss],feed_dict={X_input:X_train[b*batch_size:(b+1)*batch_size],
                                                  y_input:y_train[b*batch_size:(b+1)*batch_size]})

    loss_values_list.append(np.mean(np.array(loss_)))

    # Sometimes (if epoch%2==0), you print the loss values
    if epoch%100==0:
      # print("loss_",loss_.shape)
      # loss_ (32,)

      # Following 1D array indicates 32 number of loss values with respect to 32 number of predictions from NN

      print("loss_",np.mean(np.array(loss_)))
      # loss_ 0.46306887
      # loss_ 6.5175765e-05
      # loss_ 4.3956943e-06
      # loss_ 3.2782484e-07
      # loss_ 2.6077027e-08
      # loss_ 0.0
      # loss_ 1.8669505e-05
      # loss_ 2.2164902e-06
      # loss_ 3.0174775e-07
      # loss_ 1.117587e-08

plt.title("Loss value decreasing")
plt.xlabel("Training epoch")
plt.ylabel("Loss value")
plt.plot(loss_values_list)
plt.savefig("./loss_graph.png")
# https://github.com/youngminpark2559/test/blob/master/add_new_fc_layers_using_tensorflow/pics/loss_graph.png
plt.show()
