# Multi-scale object detection

  A Multi-scale Target Detection Algorithm Based on Deep Learning

## Composition

  multi-scale_object_detection

* `object_detection`: [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

* `slim_utils`: [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)

* `multiscalenet.py`

* `multiscalenet_train.py`: imitate [mobilenet_v1_train.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_train.py)

* `preprocess_op.py`

* `ssd_multiscalenet_feature_extractor.py`: imitate [ssd_mobilenet_v2_feature_extractor.py](https://github.com/tensorflow/models/blob/master/research/object_detection/models/ssd_mobilenet_v2_feature_extractor.py)

* `README.md`