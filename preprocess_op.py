from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from shape_utils import combined_static_and_dynamic_shape


def grid_coord(coord_x, coord_y, scope=None):
  """"Get grid coordinate.
  
  Args:
    coord_x, coord_y: A 1-D `Tensor`.
    scope: A name for the operation (optional).
  Returns:
    A 2-D `Tensor` of shape [coord_x.size*coord_y.size, 2]. 
    Has the same type as `coord_x`/`coord_y`.
  """
  with tf.name_scope(scope, 'Grid_coord', [coord_x, coord_y]):
    coord = tf.meshgrid(coord_x, coord_y)
    coord = tf.stack(coord, axis=2)
    coord = tf.reshape(coord, [-1, 2], name='coord')
  return coord


def generate_boxes(grid_size, scope=None):
  """Genetate grid boxes according to given scale size.

  Args:
  grid_size: An positive integer. The row or column of grid.
    scope: A name for the operation (optional).
  Returns:
    A `Tensor` of shape [grid_size^2, 4] and has type tf.float32.
  """
  with tf.name_scope(scope, 'Boxes_generator', [grid_size]):
    grid_size = tf.cast(grid_size, tf.int32)
    node = tf.linspace(0., 1., grid_size+1)
    ul_node = tf.slice(node, [0], [grid_size])
    ul_coord = grid_coord(ul_node, ul_node, scope='Grid_coord_ul')
    dr_node = tf.slice(node, [1], [grid_size])
    dr_coord = grid_coord(dr_node, dr_node, scope='Grid_coord_dr')
    boxes = tf.concat([ul_coord, dr_coord], 1, name='grid_boxes')
  return boxes


def get_loc_channels(boxes, size, scope=None):
  """Get location/coordinate `(x, y)` channels.

  Args:
    boxes: A 2-D `Tensor` of shape [None, 4]. Has dtype `tf.float32`.
    size: A 1-D `Tensor`/`list`/`tuple` of [height, width]. 
    scope: A name for the operation (optional).
  Returns:
    A 4-D `Tensor` of shape [None, height, width, 2]. Has same dtype as `boxes`.
  """
  with tf.name_scope(scope, 'Loc_channels_generator', [boxes, size]):
    def fn(input):
      x = tf.linspace(input[0], input[2], size[0])
      y = tf.linspace(input[1], input[3], size[1])
      x_channel = tf.transpose(tf.tile([x], [size[1], 1]), [1, 0])
      y_channel = tf.tile([y], [size[0], 1])
      return tf.stack([x_channel, y_channel], 2)

    loc_channels = tf.map_fn(fn, boxes, name='loc_channels')
    return loc_channels


def scale_preprocess(images, grid_size, crop_size, scope=None):
  """Scale preprocessing of a batch of images.

  Args:
    images: A 4-D `Tensor` of shape [batch_size, image_height, image_width, image_channels].
    grid_size: An positive integer. The row or column of grid.
    crop_size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. 
    All cropped image patches are resized to this size. The aspect ratio of the image content is not preserved. 
    Both `crop_height` and `crop_width` need to be positive.
    scope: A name for the operation (optional).
  Returns:
    A 5-D `Tensor` of shape [batch_size, num_boxes, crop_height, crop_width, image_channels+2]
  """
  with tf.name_scope(scope, 'Scale_preprocessor_%d'%(grid_size), [images, grid_size, crop_size]):
    boxes = generate_boxes(grid_size)
    boxes_shape = combined_static_and_dynamic_shape(boxes)
    box_ind = tf.zeros([boxes_shape[0]], tf.int32)
    def fn(input):
      return tf.image.crop_and_resize(input, boxes, box_ind, crop_size)
    inputs = tf.expand_dims(images, 1)
    scaled_images = tf.map_fn(fn, inputs, name='scaled_images')

    images_shape = combined_static_and_dynamic_shape(images)
    loc_channels = get_loc_channels(boxes, crop_size)
    loc_channels = tf.tile([loc_channels], [images_shape[0], 1, 1, 1, 1])
    scaled_loc_images = tf.concat([scaled_images, loc_channels], -1, name='scaled_loc_images')
    return scaled_loc_images