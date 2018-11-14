import tensorflow as tf


def generate_boxes(scale_size, scope=None):
    with tf.name_scope(scope, 'Boxes_generator', [scale_size]):
        node = tf.linspace(0, 1, scale_size+1)
        ul_node = tf.slice(node, [0], [scale_size])
        dr_node = tf.slice(node, [1], [scale_size])


def multi_scale_preprocess(image, scope=None):
    with tf.name_scope(scope, 'Multi_scale_preprocessor', [image]):
        boxes = generate_boxes(scale_size)
        tf.image.crop_and_resize(image, boxes, box_ind, crop_size)