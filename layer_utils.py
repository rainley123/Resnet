import tensorflow as tf 
import tensorflow.contrib.slim as slim

def build_block(inputs, filters, first=False, max_pool=False, stride=1):
    short_cut = inputs
    if first == True:
        net = slim.conv2d(inputs, filters, 1, stride=2)
        short_cut = slim.conv2d(short_cut, filters * 4, 1, stride=2, activation_fn=None)
    else:
        net = slim.conv2d(inputs, filters, 1, stride=stride)
        if max_pool == True:
            short_cut = slim.conv2d(short_cut, filters * 4, 1, stride=1, activation_fn=None)

    net = slim.conv2d(net, filters, 3, stride=stride)
    net = slim.conv2d(net, filters * 4, 1, stride=stride, activation_fn=None)

    net = net + short_cut
    net = tf.nn.relu(net)
    return net

def resnet101_body(inputs):
    # Conv1
    with tf.variable_scope('Conv1'):
        net = slim.conv2d(inputs, 64, 7, stride=2)

    # Conv2_x
    with tf.variable_scope('Conv2_x'):
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME')
        net = build_block(net, 64, max_pool=True)
        for i in range(2):
            net = build_block(net, 64, first=False)
    
    # Conv3_x
    with tf.variable_scope('Conv3_x'):
        net = build_block(net, 128, first=True)
        for i in range(3):
            net = build_block(net, 128, first=False)

    # Conv4_x
    with tf.variable_scope('Conv4_x'):
        net = build_block(net, 256, first=True)
        for i in range(22):
            net = build_block(net, 256, first=False)

    # Conv5_x
    with tf.variable_scope('Conv5_x'):
        net = build_block(net, 512, first=True)
        for i in range(2):
            net = build_block(net, 512, first=False)

    return net