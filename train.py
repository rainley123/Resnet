from Resnet_model import resnet101
from data_utils import parser
import tensorflow as tf 
import numpy as np
import random
import cv2
import csv
import os

# define clasify: background and car
LABELS = {
    'star': (0, 'star'),
    'noise': (1, 'noise'),
}

ANNOTATIONS = '/home/ley/Documents/futurelab/list.csv'
TRAIN_IMAGE = '/home/ley/Documents/futurelab/af2019-cv-training-20190312/'

TRAIN_TFRECORDS = './tfrecords/train*'
VAL_TFRECORDS = './tfrecords/val.tfrecords'

TRAIN_NUM = 5000
VAL_NUM = 1288

TRAIN_EVAL_INTERNAL = 250
VAL_EVAL_INTERNAL = 1
SAVE_INTERNAL = 1

BATCH_SIZE = 10
IMAGE_SIZE = [224, 224]
SHUFFLE_SIZE = 500
NUM_PARALLEL = 10

# learning rate and optimizer
OPTIMIZER = 'adam'
LEARNING_RATE_INIT = 1e-3
LEARNING_RATE_TYPE = 'exponential'
LEARNING_RATE_DECAY_STEPS = 300
LEARNING_RATE_DECAY_RATE = 0.96
LEARNING_RATE_MIN = 1e-6

# def read_file(image_path):
#     image_list = []
#     for root, dirs, files in os.walk(image_path, topdown=True):
#         for name in files:
#             basic_name = os.path.splitext(name)[0][:-2]
#             full_path = os.path.join(root, basic_name)
#             if 'jpg' in name:
#                 image_list.append(full_path)

#     image_list = list(set(image_list))
#     random.shuffle(image_list)

#     train_list = image_list[0 : 5000]
#     val_list = image_list[5000 : -1]
#     return train_list, val_list

# def get_info(annotation, csv_file):
#     boxes = []
#     labels = []

#     for obj in csv_file:
#         if obj[0] == annotation:
#             center_x = float(obj[1])
#             center_y = float(obj[2])
#             xmin = center_x - 4.
#             xmax = center_x + 4.
#             ymin = center_y - 4.
#             ymax = center_y + 4.
#             boxes.append([xmin, ymin, xmax, ymax])
#             if (obj[3] == 'noise' or obj[3] == 'ghost' or obj[3] == 'pity'):
#                 labels.append(int(LABELS['noise'][0]))
#             else:
#                 labels.append(int(LABELS['star'][0]))
#     return labels, boxes

# def get_batch(image_list, batch_size):
#     image_batch = []
#     boxes_batch = []
#     labels_batch = []
    
#     index_list = [random.randint(0, len(image_list) - 1) for _ in range(10)]
#     for index in index_list:
#         image_name = image_list[index]
#         annotation_name = image_name.split('/')[-1]

#         image_a = cv2.imread(image_name + '_a.jpg')[:, :, 0]
#         image_b = cv2.imread(image_name + '_b.jpg')[:, :, 0]
#         image_c = cv2.imread(image_name + '_c.jpg')[:, :, 0]
#         image = np.stack([image_a, image_b, image_c], axis=-1)
#         image_batch.append(image)

#         csv_file = csv.reader(open(ANNOTATIONS, 'r'))
#         labels, boxes = get_info(annotation_name, csv_file)
#         labels_batch.append(labels)
#         boxes_batch.append(boxes)

#     # Preprocess
#     pre_image = []
#     pre_boxes = []
#     pre_labels = []
#     for i in range(batch_size):
#         image, boxes, labels = preprocess(image_batch[i], np.array(boxes_batch[i], dtype=np.float32), np.array(labels_batch[i], dtype=np.int64), IMAGE_SIZE, 'train')
#         pre_image.append(image)
#         pre_boxes.append(boxes)
#         pre_labels.append(labels)
#     pre_image = tf.reshape(pre_image, shape=[batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
#     pre_boxes = tf.reshape(pre_boxes, shape=[batch_size, 1, 2])
#     pre_labels = tf.reshape(pre_labels, shape=[batch_size, 1])
#     return pre_image, pre_boxes, pre_labels

with tf.Graph().as_default():
    # input_image = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    # input_boxes = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1, 2])
    # input_labels = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE, 1])
    # is_training = tf.placeholder(tf.bool)

    # y_true = [input_boxes[:, :, 0], input_boxes[:, :, 1], input_labels[:, 0]]

    train_files = tf.train.match_filenames_once(TRAIN_TFRECORDS)
    train_dataset = tf.data.TFRecordDataset(train_files, buffer_size=5)
    train_dataset = train_dataset.map(lambda x : parser(x, IMAGE_SIZE, 'train'), num_parallel_calls=NUM_PARALLEL)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    val_files = [VAL_TFRECORDS]
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(lambda x : parser(x, IMAGE_SIZE, 'val'), num_parallel_calls=NUM_PARALLEL)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    # create a public iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

    image, boxes, labels = iterator.get_next()
    y_true = [boxes[:, :, 0], boxes[:, :, 1], labels[:, 0]]

    image.set_shape([None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    # define the yolo_v3 model
    is_training = tf.placeholder(tf.bool)

    resnet_model = resnet101()
    with tf.variable_scope('resnet101'):
        features = resnet_model.forward(image, is_training)
    loss = resnet_model.compute_loss(features, y_true)

    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    # define the learning rate and optimizer
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step, decay_steps=LEARNING_RATE_DECAY_STEPS,
                                 decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True, name='exponential_learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # summary the loss and learning_rate
    tf.summary.scalar('total_loss', loss[0])
    tf.summary.scalar('loss_xy', loss[1])
    tf.summary.scalar('loss_class', loss[2])
    tf.summary.scalar('learning_rate', learning_rate)

    # saver_to_restore = tf.train.Saver()

    # average model 
    ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
    ema_op = ema.apply(tf.trainable_variables())

    # update the BN vars
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = optimizer.minimize(loss[0], global_step=global_step)
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

    # set session config
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.85

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        # saver_to_restore.restore(sess, RESTORE_PATH)
        saver = tf.train.Saver()

        write_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter("./log/train", sess.graph)
        writer_val = tf.summary.FileWriter("./log/val", sess.graph)

        print('\n------------- start to train --------------\n')
        
        # train_list, val_list = read_file(TRAIN_IMAGE)
        for epoch in range(2000):
            sess.run(iterator.make_initializer(train_dataset))
            while(True):
                try:
                    # image_train, boxes_train, labels_train,  = get_batch(train_list, BATCH_SIZE)
                    _, summary, loss_, global_step_, learn_rate_ = sess.run([train_op, write_op, loss, global_step, learning_rate], 
                                        feed_dict={is_training: True})

                    writer_train.add_summary(summary, global_step=global_step_)
                    info = "global_step: {}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_class: {:.3f}".format(
                    global_step_, loss_[0], loss_[1], loss_[2])
                    print(info)
                except tf.errors.OutOfRangeError:
                    break