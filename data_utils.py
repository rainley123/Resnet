import tensorflow as tf 
import numpy as np 
import random
import cv2

def flip_left_right(image, boxes, labels):
    width = tf.cast(tf.shape(image)[1], tf.float32)
    image  = tf.image.flip_left_right(image)

    xmin = 0 - boxes[:, 2] + width
    ymin = boxes[:, 1]
    xmax = 0 - boxes[:, 0] + width
    ymax = boxes[:, 3]
    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    return image, boxes, labels

def flip_down_up(image, boxes, labels):
    height = tf.cast(tf.shape(image)[0], tf.float32)
    image  = tf.image.flip_up_down(image)

    xmin = boxes[:, 0]
    ymin = 0 - boxes[:, 3] + height
    xmax = boxes[:, 2]
    ymax = 0 - boxes[:, 1] + height
    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    return image, boxes, labels

def rotate_image(image, boxes, labels):
    image = np.array(image, dtype=np.float32)
    height = int(image.shape[0])
    width = int(image.shape[1])
    center = (width / 2, height / 2)
    angel = random.randint(0, 180)
    M = cv2.getRotationMatrix2D(center, angel, 1)
    image = cv2.warpAffine(image, M, (width, height))
    return image, boxes, labels

def distort_color(image, boxes, labels):
    sequence = [0, 1, 2, 3]
    random.shuffle(sequence)
    for i in sequence:
        if i == 0:
            image = tf.image.random_brightness(image, max_delta=32./255)
        if i == 1:
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        if i == 2:
            image = tf.image.random_hue(image, max_delta=0.2)
        if i == 3:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, boxes, labels

def median_filter(image):
    image_a, image_b, image_c = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    blur_a = cv2.medianBlur(np.array(image_a), 3)
    blur_b = cv2.medianBlur(np.array(image_b), 3)
    blur_c = cv2.medianBlur(np.array(image_c), 3)
    image = np.stack([blur_a, blur_b, blur_c], axis=-1)
    return image

def crop(image, boxes, labels, min_object_covered=0.5, aspect_ratio_range=[0.5, 2.0], area_range=[0.3, 1.0]):
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)
    bboxes = tf.stack([ymin/h, xmin/w, ymax/h, xmax/w], axis=1)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    begin, size, dist_boxes = tf.image.sample_distorted_bounding_box(
                                    tf.shape(image),
                                    bounding_boxes=tf.expand_dims(bboxes, axis=0),
                                    min_object_covered=min_object_covered,
                                    aspect_ratio_range=aspect_ratio_range,
                                    area_range=area_range,
                                    max_attempts=50)
    # NOTE dist_boxes with shape: [ymin, xmin, ymax, xmax] and in values in range(0, 1)
    # Employ the bounding box to distort the image.
    croped_box = [dist_boxes[0,0,1]*w, dist_boxes[0,0,0]*h, dist_boxes[0,0,3]*w, dist_boxes[0,0,2]*h]

    croped_xmin = tf.clip_by_value(xmin, croped_box[0], croped_box[2])-croped_box[0]
    croped_ymin = tf.clip_by_value(ymin, croped_box[1], croped_box[3])-croped_box[1]
    croped_xmax = tf.clip_by_value(xmax, croped_box[0], croped_box[2])-croped_box[0]
    croped_ymax = tf.clip_by_value(ymax, croped_box[1], croped_box[3])-croped_box[1]

    image = tf.slice(image, begin, size)
    boxes = tf.stack([croped_xmin, croped_ymin, croped_xmax, croped_ymax], axis=1)

    return image, boxes, labels

def resize_image_and_correct_boxes(image, boxes, labels, image_size):
    origin_image_size = tf.to_float(tf.shape(image)[0:2])
    def w_long():
        new_w = image_size[1]
        new_h = tf.to_int32(origin_image_size[0] / origin_image_size[1] * image_size[1])
        return [new_h, new_w]

    def h_long():
        new_h = image_size[0]
        new_w = tf.to_int32(origin_image_size[1] / origin_image_size[0] * image_size[0])  
        return [new_h, new_w]

    new_size = tf.cond(tf.less(origin_image_size[0] / image_size[0], origin_image_size[1] / image_size[1]), 
                        w_long, h_long)

    image = tf.image.resize_images(image, new_size)
    offset_h = tf.to_int32((image_size[0] - new_size[0]) / 2)
    offset_w = tf.to_int32((image_size[1] - new_size[1]) / 2)
    image = tf.image.pad_to_bounding_box(image, offset_h, offset_w, image_size[0], image_size[1])
    
    # correct the boxes
    xmin = tf.clip_by_value(boxes[:, 0] / origin_image_size[1], 0.0, 1.0) * tf.to_float(new_size[1]) + tf.to_float(offset_w)
    ymin = tf.clip_by_value(boxes[:, 1] / origin_image_size[0], 0.0, 1.0) * tf.to_float(new_size[0]) + tf.to_float(offset_h)
    xmax = tf.clip_by_value(boxes[:, 2] / origin_image_size[1], 0.0, 1.0) * tf.to_float(new_size[1]) + tf.to_float(offset_w)
    ymax = tf.clip_by_value(boxes[:, 3] / origin_image_size[0], 0.0, 1.0) * tf.to_float(new_size[0]) + tf.to_float(offset_h)

    # if the object is not in the dist_box, just remove it 
    mask = tf.logical_not(tf.logical_or(tf.equal(xmin, xmax), tf.equal(ymin, ymax)))        
    xmin = tf.boolean_mask(xmin, mask)
    ymin = tf.boolean_mask(ymin, mask)
    xmax = tf.boolean_mask(xmax, mask)
    ymax = tf.boolean_mask(ymax, mask)
    labels = tf.boolean_mask(labels, mask)

    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return image, boxes, labels

def data_augmentation(image, boxes, labels):
    if random.randint(0, 1) == 0:
        image, boxes, labels = flip_left_right(image, boxes, labels)
    if random.randint(0, 1) == 0:
        image, boxes, labels = flip_down_up(image, boxes, labels)
    if random.randint(0, 1) == 0:
        image, boxes, labels = distort_color(image, boxes, labels)
    image, boxes, labels = crop(image, boxes, labels)

    return image, boxes, labels

def preprocess(image, boxes, labels, image_size, mode):
    if len(image.get_shape().as_list()) != 3:
        raise ValueError('Input image must have 3 shapes H W C')
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # data augmentation for train data
    if mode == 'train':  
        image, boxes, labels = data_augmentation(image, boxes, labels)

    image, boxes, labels = resize_image_and_correct_boxes(image, boxes, labels, image_size)

    # image, boxes, labels = sub_mean(image, boxes, labels)
    image = tf.py_function(median_filter, inp=[image], Tout=[tf.float32])
    image = tf.reshape(image, [image_size[0], image_size[1], 3])

    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.
    boxes = tf.stack([center_x, center_y], axis=-1)

    return image, boxes, labels

def parser(serialized_example, image_size, mode):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_a' : tf.FixedLenFeature([], dtype = tf.string),
            'image_b' : tf.FixedLenFeature([], dtype = tf.string),
            'image_c' : tf.FixedLenFeature([], dtype = tf.string),
            'boxes' : tf.FixedLenFeature([], dtype = tf.string),
            'labels' : tf.FixedLenFeature([], dtype = tf.string),
        })

    image_a = features['image_a']
    image_b = features['image_b']
    image_c = features['image_c']
    boxes = features['boxes']
    labels = features['labels']

    image_a = tf.image.decode_jpeg(image_a)
    image_b = tf.image.decode_jpeg(image_b)
    image_c = tf.image.decode_jpeg(image_c)
    
    image_a = tf.squeeze(image_a)
    image_a.set_shape([None, None])
    image_b = tf.squeeze(image_b)
    image_b.set_shape([None, None])
    image_c = tf.squeeze(image_c)
    image_c.set_shape([None, None])

    image = tf.stack([image_a, image_b, image_c], axis=-1)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    boxes = tf.decode_raw(boxes, tf.float32)
    boxes = tf.reshape(boxes, shape=[-1, 4])

    labels = tf.decode_raw(labels, tf.int64)
    labels = tf.reshape(labels, shape=[-1])
   
    return preprocess(image, boxes, labels, image_size, mode)