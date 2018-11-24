"""Multiscalenet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

import preprocess_op
from shape_utils import combined_static_and_dynamic_shape
from slim_utils.nets.mobilenet import mobilenet
from slim_utils.nets.mobilenet.mobilenet import safe_arg_scope
from slim_utils.nets.mobilenet import mobilenet_v2


CLS_MODEL_MAP = {
  'mobilenet_v2': mobilenet_v2.mobilenet
} 

CLS_MODEL_ARG_SCOPE_MAP = {
  'mobilenet_v2': mobilenet_v2.training_scope
} 


class Multiscalenet(object):
  """Multiscalenet definition."""
  def __init__(self,
               num_classes=1001,
               model_name='mobilenet_v2',
               is_training=True,
               scale_list=[1],
               crop_size=[320,320],
               depth_multiplier=1.0,
               scope='Multiscalenet'):
    """Multiscalenet model for detection.

    Args:
      num_classes: Number of classes.
      model_name: Which model should be added.
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      scale_list: A list indicating what scales the input_tensor should include.
      depth_multiplier: The multiplier applied to scale number of channels
        in each layer.
      scope: Scope of the operator.
    Raises:
      ValueError: If scale_list is not of type list.
      ValueError: If model_name is not in `CLS_MODEL_MAP`.
    """
    if (not isinstance(scale_list, list)) or (len(scale_list) == 0):
      raise ValueError('scale_list must be of type `list` and not None.')

    if model_name not in CLS_MODEL_MAP:
      raise ValueError('Unknown model: {}'.format(model_name))
    
    self._model_fn = CLS_MODEL_MAP[model_name]
    self._model_arg_scope = CLS_MODEL_ARG_SCOPE_MAP[model_name]
    self._num_classes = num_classes
    self._is_training = is_training
    self._scale_list = scale_list
    self._crop_size = crop_size
    self._depth_multiplier = depth_multiplier
    self._scope = scope

  @property
  def num_classes(self):
    return self._num_classes

  def multiscale_preprocess(self, input_tensor):
    """Multiscalenet preprocess module.
    
    Args:
      input_tensor: Input batch images.
    Returns:
      multiscaled_layer: The output of the Multiscalenet preprocess module.
      scaled_layers: Some layers in the Multiscalenet preprocess module.
    """
    with tf.variable_scope(self._scope):
      input_shape = combined_static_and_dynamic_shape(input_tensor)

      scaled_layers = {}
      with slim.arg_scope(self._multiscale_training_scope(self._is_training)):
        for scale in self._scale_list:
          scope = 'scaled_layer_%d'%(scale)
          with tf.variable_scope(scope):
            scaled_tensor = preprocess_op.scale_preprocess(input_tensor, scale, self._crop_size)
            if scale == 1:
              net = tf.squeeze(scaled_tensor, [1])
              net = slim.conv2d(net, 32, 3, 1)
            else:
              net = slim.conv3d(scaled_tensor, 16, 3, 1)
              net = tf.reshape(net, [input_shape[0], self._crop_size[0], self._crop_size[1], (scale^2)*16])
              net = slim.conv2d(net, 32, 3, 1)
            scaled_layers[scope] = net
      multiscaled_layer = tf.concat(list(scaled_layers.values()), -1, name='multiscaled_layer')
      scaled_layers['multiscaled_layer'] = multiscaled_layer
    return multiscaled_layer, scaled_layers

  def build_model(self, input):
    """"Bulid backbone model."""
    multiscale_net, scaled_layers = self.multiscale_preprocess(input)
    with slim.arg_scope(
        self._model_arg_scope(is_training=self._is_training, bn_decay=0.9997)):
      logits, end_points = self._model_fn(
          multiscale_net,
          self._num_classes,
          depth_multiplier=self._depth_multiplier)
    end_points.update(scaled_layers)
    return logits, end_points


  def _multiscale_training_scope(self, 
                                is_training=True,
                                weight_decay=0.00004,
                                stddev=0.09,
                                bn_decay=0.997):
    """Defines Multiscalenet training scope.

    Args:
      is_training: if set to False this will ensure that all customizations are
        set to non-training mode. This might be helpful for code that is reused
        across both training/evaluation, but most of the time training_scope with
        value False is not needed. If this is set to None, the parameters is not
        added to the batch_norm arg_scope.

      weight_decay: The weight decay to use for regularizing the model.
      stddev: Standard deviation for initialization, if negative uses xavier.
      bn_decay: decay for the batch norm moving averages (not set if equals to
        None).

    Returns:
      An argument scope to use via arg_scope.
    """
    batch_norm_params = {
        'decay': bn_decay,
        'is_training': is_training
    }
    if stddev < 0:
      weight_intitializer = slim.initializers.xavier_initializer()
    else:
      weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv layers.
    with slim.arg_scope(
        [slim.conv2d, slim.conv3d, slim.separable_conv2d],
        weights_initializer=weight_intitializer,
        normalizer_fn=slim.batch_norm), \
        safe_arg_scope([slim.batch_norm], **batch_norm_params), \
        slim.arg_scope([slim.conv2d, slim.conv3d], \
                       weights_regularizer=slim.l2_regularizer(weight_decay)), \
        slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
      return s