import tensorflow as tf
from tensorflow.contrib import slim

import preprocess_op
from shape_utils import combined_static_and_dynamic_shape


class Multiscalenet(object):
  def __init__(self,
               is_training,
               num_classes,
               scale_list=[1],
               crop_size=[320,320],
               depth_multiplier=1.0,
               scope='Multiscalenet'):
    """Multiscalenet model for detection.

    Args:
      is_training: A boolean indicating whether the training version of the
      computation graph should be constructed.
      num_classes: Number of classes.
      scale_list: A list indicating what scales the input_tensor should include.
      depth_multiplier: The multiplier applied to scale number of channels
      in each layer.
      scope: Scope of the operator.
    Raises:
      ValueError: If scale_list is not of type list.
    """
    if (not isinstance(scale_list, list)) or (len(scale_list) == 0):
      raise ValueError('scale_list must be of type `list` and not None.')

    self._is_training = is_training
    self._num_classes = num_classes
    self._scale_list = scale_list
    self._crop_size = crop_size
    self._depth_multiplier = depth_multiplier
    self._scope = scope

  @property
  def num_classes(self):
    return self._num_classes

  def build_model(self, input_tensor):
    input_shape = combined_static_and_dynamic_shape(input_tensor)

    scaled_layers = []
    for scale in self._scale_list:
      scaled_tensor = preprocess_op.scale_preprocess(input_tensor, scale, self._crop_size)
      with tf.variable_scope('scaled_layer_%2d'%(scale)) as scope:
        if scale == 1:
          net = tf.squeeze(scaled_tensor, [1])
          net = slim.convolution2d(net, 32, 3, 2, scope)
        else:
          net = slim.convolution3d(scaled_tensor, 16, 3, 2, scope)
          net = tf.reshape(net, [input_shape[0], self._crop_size[0], self._crop_size[1], -1])
          net = slim.convolution2d(net, 32, 3, 1, scope)
        scaled_layers.append(net)
    multiscaled_layer = tf.concat(scaled_layers, -1)
    multiscaled_layer = slim.convolution2d(multiscaled_layer, 16, 3, 1)
    multiscaled_layer = tf.identity(multiscaled_layer, name='multiscaled_layer')