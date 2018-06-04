import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math
from Lab10.faceUtil import *
import time



# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
face = getFaceData()
# image_size = 28
image_size = [112, 92]
num_classes = 40



def display(val):
    image = face.train.images[val].reshape([image_size[0], image_size[1]])
    label = face.train.labels[val].argmax()
    plt.title('Training: %d  Label: %d' % (val, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def createConvolutionLayer(x_input, kernel_size, features, depth):
    # createConvolutionLayer generates a convolution layer in the session graph
    # by assigning weights, biases, convolution and relu function
    #
    # x_input - output from the previous layer
    # kernel_size - size of the feature kernels
    # depth - number of feature kernels
    #
    # returns convolution layer in graph
    #
    print("conv: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, features, depth],
                              initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0))
    print("shape:" + str(x_input.get_shape()))
    print("shape:" + str(weights.get_shape()))
    convolution = tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME')
    print("shape:" + str(convolution.get_shape()))
    added = tf.nn.bias_add(convolution, biases)

    return tf.nn.relu(added), convolution, weights


def createFullyConnectedLayer(x_input, width):
    # createFullyConnectedLayer generates a fully connected layer in the session graph
    #
    # x_input - output from previous layer
    # width - width of the layer (eg for a 10 class output you need to end with a 10 width layer
    #
    # returns fully connected layer in graph
    #
    print("fc: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[x_input.get_shape()[1], width],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[width], initializer=tf.constant_initializer(0))

    matrix_multiply = tf.matmul(x_input, weights)

    return tf.nn.bias_add(matrix_multiply, biases)


def createSoftmaxLayer(x_input, width):
    # createSoftmaxLayer generates a softmax layer in the session graph
    #
    # x_input - output from previous layer
    # width - width of the layer (eg for a 10 class output you need to end with a 10 width layer
    #
    # returns softmax layer in graph
    #
    print("softmax: input size: " + str(x_input.get_shape()))
    weights = tf.get_variable('weights', shape=[x_input.get_shape()[1], width],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[width], initializer=tf.constant_initializer(0))

    matrix_multiply = tf.matmul(x_input, weights)

    return tf.nn.softmax(tf.nn.bias_add(matrix_multiply, biases))


def createLinearRectifier(x_input):
    # createLinearRectifier generates a ReLu in the session graph
    #
    # The reason this exists is due to the last fully connected layer not needing a relu while others do
    # x_input - output from previous layer
    # width - width of the layer
    #
    # returns ReLu in graph
    #

    return tf.nn.relu(x_input)

def createPoolingLayer(x_input, kernel_size):
    # createPoolingLayer generates a pooling layer in the session graph
    #
    # The reason this exists is due to the last fully connected layer not needing a relu while others do
    # x_input - output from previous layer
    # kernel_size - size of the kernel
    #
    # returns pooling layer in graph
    #
    print("pool: input size: " + str(x_input.get_shape()))
    return tf.nn.max_pool(x_input, ksize=[1, kernel_size, kernel_size, 1], strides=[1,kernel_size,kernel_size, 1], padding='SAME')


def createNetwork(x_input, is_training):
    with tf.variable_scope('conv1'):
        print("shape:" + str(x_input.get_shape()))
        convolution_layer1, just_conv1, weights1 = createConvolutionLayer(x_input, 5, 1, 32)
        print("shape:" + str(convolution_layer1.get_shape()))
        pooling_layer1 = createPoolingLayer(convolution_layer1, 1)
    with tf.variable_scope('conv2'):
        convolution_layer2, just_conv2, weights2 = createConvolutionLayer(pooling_layer1, 5, 32, 64)
        pooling_layer1 = createPoolingLayer(convolution_layer2, 1)
        pooling_layer1_shape = pooling_layer1.get_shape().as_list()
        pooling_layer1_flattened = tf.reshape(pooling_layer1, [-1, pooling_layer1_shape[1] * pooling_layer1_shape[2] *
                                                               pooling_layer1_shape[3]])
    with tf.variable_scope('fc1'):
        fully_connected_layer1 = createFullyConnectedLayer(pooling_layer1_flattened, 1024)
        fully_connected_relu1 = createLinearRectifier(fully_connected_layer1)
        # fully_connected_relu1 = tf.cond(is_training, lambda: tf.nn.dropout(fully_connected_relu1, keep_prob=0.5), lambda: fully_connected_relu1)
    with tf.variable_scope('softmax'):
        output = createSoftmaxLayer(fully_connected_relu1, num_classes)

    return output, convolution_layer1, convolution_layer2, just_conv1, just_conv2, weights1, weights2

# def createNetwork(x_input, is_training):
#     #Use 3 convolutional layers now
#     with tf.variable_scope('conv1'):
#         print("input shape:" + str(x_input.get_shape()))
#         convolution_layer1, just_conv1, weights1 = createConvolutionLayer(x_input, 5, 1, 32)
#         print("conv1 shape:" + str(convolution_layer1.get_shape()))
#         pooling_layer1 = createPoolingLayer(convolution_layer1, 2)
#         print("pool1 shape:" + str(pooling_layer1.get_shape()))
#     with tf.variable_scope('conv2'):
#         convolution_layer2, just_conv2, weights2 = createConvolutionLayer(pooling_layer1, 5, 32, 64)
#         print("conv2 shape:" + str(convolution_layer2.get_shape()))
#         pooling_layer1 = createPoolingLayer(convolution_layer2, 2)
#         print("pool2 shape:" + str(pooling_layer1.get_shape()))
#
#
#     with tf.variable_scope('conv3'):
#         convolution_layer3, just_conv3, weights3 = createConvolutionLayer(pooling_layer1, 5, 64, 64)
#         pooling_layer1 = createPoolingLayer(convolution_layer3, 2)
#         print("pool3 shape:" + str(pooling_layer1.get_shape()))
#
#         pooling_layer1_shape = pooling_layer1.get_shape().as_list()
#         pooling_layer1_flattened = tf.reshape(pooling_layer1, [-1, pooling_layer1_shape[1] * pooling_layer1_shape[2] *
#                                                                pooling_layer1_shape[3]])
#     with tf.variable_scope('fc1'):
#         fully_connected_layer1 = createFullyConnectedLayer(pooling_layer1_flattened, 1024)
#         fully_connected_relu1 = createLinearRectifier(fully_connected_layer1)
#         # fully_connected_relu1 = tf.cond(is_training, lambda: tf.nn.dropout(fully_connected_relu1, keep_prob=0.5), lambda: fully_connected_relu1)
#     with tf.variable_scope('softmax'):
#         output = createSoftmaxLayer(fully_connected_relu1, num_classes)
#
#     return output, convolution_layer1, convolution_layer2, convolution_layer3, just_conv1, just_conv2, just_conv3, weights1, weights2, weights3


graph = tf.Graph()

with graph.as_default():
    x_input = tf.placeholder(tf.float32, shape=[None, image_size[0]* image_size[1]], name='x_input')
    y_output = tf.placeholder(tf.int64, shape=[None], name='y_output')
    is_training = tf.placeholder(tf.bool)

    # learning rate
    learning_rate = 0.0001

    # get model
    # x_image = tf.reshape(x_input, [-1, 28, 28, 1])
    x_image = tf.reshape(x_input, [-1, 112, 92, 1])
    prediction_output, convr1, convr2, conv1, conv2, weights1, weights2 = createNetwork(x_image, is_training)

    # correct_prediction = tf.equal(tf.argmax(y_output, axis=-1), tf.argmax(prediction_output, axis=-1))
    correct_prediction = tf.equal(y_output, tf.argmax(prediction_output, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # loss_function = tf.reduce_mean(-tf.reduce_sum(y_output * tf.log(prediction_output), reduction_indices=[1]))
    # loss = y_output * tf.log(prediction_output)
    # print("shape of loss is  " + str(loss.shape))

    onehot_labels = tf.one_hot(indices=tf.cast(y_output, tf.int32), depth=40)
    loss_function = tf.reduce_mean(-tf.reduce_sum(onehot_labels * tf.log(prediction_output)))

    # optimization method
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)


model_name = "facesTuning"
model_filename = model_name + "Model.ckpt"
model_directory = os.getcwd() + "/" + model_name
model_path = model_directory + "/" + model_filename


batch_size = 64
start = time.time()
train_loss_list = []
valid_loss_list = []
time_list = []
epoch_list = []
print("TRAINING: " + model_name)
with tf.Session(graph = graph) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists(model_directory):
        load_path = saver.restore(session, model_path)
    for i in range(100):
        # batch = mnist.train.next_batch(batch_size)
        subdata, sublabels = getBatch(face.train.images, face.train.labels, batch_size)
        # batch = np.reshape(batch, (-1, 28, 28, 1))
        if i%10 == 0:
            feed_dict = {x_input:subdata, y_output:sublabels-1, is_training: False}
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            train_loss = session.run(loss_function, feed_dict=feed_dict)
            # print(session.run(y_output, feed_dict={x_input: face.test.images}))
            # print(correct_prediction.eval())
            print("step " + str(i) + " Train Acc: " + str(train_accuracy) + " Train Loss: " + str(train_loss))
        # Train system
        session.run([optimizer], feed_dict={x_input: subdata, y_output:sublabels-1, is_training: True})

    batch_size = 80
    test_loss = session.run(loss_function, feed_dict = {x_input: face.test.images, y_output: face.test.labels-1, is_training: False})
    test_accuracy = accuracy.eval(feed_dict={x_input: face.test.images, y_output: face.test.labels-1, is_training: False})

    # test_batch = getBatch(face.test.images, face.test.labels, batch_size)
    # prediction = prediction_output.eval(feed_dict={x_input: test_batch[0], y_output: test_batch[1]-1, is_training: False})
    # predicted_ys = correct_prediction.eval(feed_dict={x_input: test_batch[0], y_output: test_batch[1]-1, is_training: False})
    # indices = [c for c, x in enumerate(predicted_ys) if x == False]
    # print(indices)
    # plt.figure(2, figsize=(10, 10))
    # columns = 8
    # rows = 10
    # fig, ax_array = plt.subplots(rows, columns, squeeze=False)
    # for i, ax_row in enumerate(ax_array):
    #     for j, axes in enumerate(ax_row):
    #         axes.imshow(face.test.images, cmap='gray')
    #         # curr = i*rows+j
    #         # if (curr) in indices:
    #         #     axes.text(24, 25, '%d' % test_batch[1][curr]-1, fontsize=5, color='red')
    #         # else:
    #         #     axes.text(24, 25, '%d' % test_batch[1][curr] - 1, fontsize=5, color='green')
    # fig.canvas.set_window_title("Testing Face Classification")
    # plt.tight_layout()
    # plt.show()



    # print(prediction.eval(feed_dict={x_input: face.test.images}))
    # test_result = session.run(, feed_dict={x_input: face.test.images, y_output: face.test.labels-1, is_training: False})
    # print("Actual Labels are" + str(test_result))

    print("Final: " " Test Acc: " + str(test_accuracy) + " Test Loss: " + str(test_loss))
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    save_path = saver.save(session, model_path)

    f, ax = plt.subplots(2, 3, figsize=(7, 7))

    test_batch = getBatch(face.test.images, face.test.labels, batch_size)
    prediction = prediction_output.eval(feed_dict={x_input: test_batch[0], y_output: test_batch[1]-1, is_training: False})
    predicted_ys = correct_prediction.eval(feed_dict={x_input: test_batch[0], y_output: test_batch[1]-1, is_training: False})
    print("correct predictions are " + str(predicted_ys))

    indices = [c for c, x in enumerate(predicted_ys) if x == False]
    np.random.seed()
    test_data  = np.asarray(test_batch[0], dtype=np.float32)  # Returns np.array
    ri = np.random.choice(len(test_data), 8, replace=False)
    selectedFalse = indices[:3]
    # for i in range(1):
    for j in range(3):
        ci = ri[j]


        # Correct example
        ax[0][j].imshow(np.reshape(test_data[ci, :], (112, 92)), cmap='gray')
        ax[0][j].axes.get_xaxis().set_visible(False)
        ax[0][j].axes.get_yaxis().set_visible(False)
        ax[0][j].text(24, 25, 'Correct: ' + '%d' % test_batch[1][ci], fontsize=15, color='cyan')

        # Incorrect example
        ax[1][j].imshow(np.reshape(test_data[selectedFalse[j], :], (112, 92)), cmap='gray')
        ax[1][j].axes.get_xaxis().set_visible(False)
        ax[1][j].axes.get_yaxis().set_visible(False)
        ax[1][j].text(24, 25, 'Wrong: ' + '%d' % selectedFalse[j], fontsize=15, color='red')
        # ax[i+1][j].text(2, 25, '%d' % predicted_ys[ww], fontsize=12, color='red')



            # plt.imshow(np.reshape(face.train.images[77], [112, 92]))
#
#
def displayMaps(layer, image, session):
    feature_maps = session.run(layer, feed_dict={x_input:image, is_training: False})
    filters = feature_maps.shape[3]
    plt.figure(2, figsize=(20,20))
    columns = 6
    rows = math.ceil(filters / columns) + 1
    for i in range(filters):
        plt.subplot(rows, columns, i+1)
        plt.title('Feature Map ' + str(i))
        plt.axis('off')
        plt.imshow(feature_maps[0,:,:,i], interpolation="nearest", cmap="gray")
#
#
def displayFilters(weights, session):
    weights = weights.eval()
    filters = weights.shape[3]
    plt.figure(3, figsize=(10,10))
    columns = 6
    rows = math.ceil(filters / columns) + 1
    for i in range(filters):
        plt.subplot(rows, columns, i+1)
        plt.title('Filter ' + str(i))
        plt.axis('off')
        plt.imshow(np.reshape(weights[:,:,:,i], [5,5]), interpolation="nearest", cmap="gray")
#
#
#
#
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer()
    load_path = saver.restore(session, model_path)
    displayMaps(convr2, np.reshape(face.train.images[78], [1, 10304]), session)

#
# print(weights1.get_shape())
#
# with tf.Session(graph = graph) as session:
#     tf.global_variables_initializer()
#     load_path = saver.restore(session, model_path)
#     print(weights2.get_shape())
#
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer()
    load_path = saver.restore(session, model_path)
    displayFilters(weights1, session)