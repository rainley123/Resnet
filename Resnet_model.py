import tensorflow as tf 
import tensorflow.contrib.slim as slim
from layer_utils import resnet101_body

class resnet101(object):

    def __init__(self, batch_norm_decay=0.999):
        self.batch_norm_decay = batch_norm_decay
    
    def forward(self, inputs, is_training=False, reuse=False):
        # The input img_size, form: [height, weight]
        self.img_size = tf.shape(inputs)[1:3]

        # Set the batch norm params
        batch_norm_param = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'fused': None,
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_param,
                                biases_initializer=None):
                with tf.variable_scope('resnet_body'):
                    net = resnet101_body(inputs) 
                
                with tf.variable_scope('resnet_fc'):
                    net = slim.avg_pool2d(net, 7, stride=1)
                    net = tf.reshape(net, [-1, 2048])
                    features = slim.fully_connected(net, 4)
        return features

    def compute_loss(self, features, y_true):
        N = tf.cast(tf.shape(features)[0], tf.float32)
        pred_x, pred_y, pred_class = tf.split(features, [1, 1, 2], axis=-1)
        true_x, true_y, true_label = y_true

        pred_xy = tf.concat([pred_x, pred_y], axis=-1)
        true_xy = tf.concat([true_x, true_y], axis=-1)

        loss_xy = tf.reduce_sum(tf.square(true_xy - pred_xy)) / N 
        loss_class = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(true_label, 2), logits=pred_class)) / N 

        loss_total = loss_xy + loss_class
        return loss_total, loss_xy, loss_class


