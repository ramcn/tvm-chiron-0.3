from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
import tensorflow as tf
from distutils.dir_util import copy_tree
from chiron_input import read_raw_data_sets
from cnn import getcnnfeature
from cnn import getcnnlogit
from rnn import rnn_layers_one_direction
import time,os
import numpy as np
import subprocess
from tensorflow.contrib.cluster_resolver import TPUClusterResolver


# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))


# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 100, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_string("cache_dir", "/uufs/chpc.utah.edu/common/home/u1142888/Chiron-0.3/output_cache", "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("sequence_len", 300, "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("k_mer", 1, "Number of shards (TPU chips).")
tf.flags.DEFINE_string("data_dir", "/uufs/chpc.utah.edu/common/home/u1142888/Chiron-0.3/new_out/", "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS


def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
    batch_x = np.array(list(features['x']))
    batch_x_tf = tf.convert_to_tensor(batch_x, dtype=tf.float32)
    indexs,values,shape = labels['y']
    y = tf.SparseTensor(indexs,values,shape)

    cnn_feature = getcnnfeature(batch_x_tf, training = True)
    feashape = cnn_feature.get_shape().as_list()
    ratio = FLAGS.sequence_len/feashape[1]
    logits = getcnnlogit(cnn_feature)

    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(y,logits, labels['seq_len'],ctc_merge_repeated = True,time_major = False))
    adam = tf.train.AdamOptimizer(1e-2)
    if FLAGS.use_tpu:
      adam = tf.contrib.tpu.CrossShardOptimizer(adam) # TPU change 1
    train_op = adam.minimize(ctc_loss, name="train_op", global_step=tf.train.get_global_step())
    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=ctc_loss, train_op=train_op)

def input_fn(params):
  del params
  batch_size = FLAGS.batch_size

  def parser(params):
    train_ds = read_raw_data_sets(FLAGS.data_dir,FLAGS.cache_dir,FLAGS.sequence_len,k_mer = FLAGS.k_mer)
    batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size)
    x={"x": batch_x }
    y={ "y": batch_y ,"seq_len": seq_len }
    return x, y

  dataset = dataset.map(parser, num_parallel_calls=batch_size)
  dataset = dataset.prefetch(4 * batch_size).cache().repeat()
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1)
  return dataset


def main(argv):
 del argv  # Unused.
 tf.logging.set_verbosity(tf.logging.INFO)

 if(FLAGS.use_tpu):
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir},
      config=run_config)
 else:
  estimator = tf.contrib.tpu.TPUEstimator( model_fn=model_fn, config=tf.contrib.tpu.RunConfig(), use_tpu=FLAGS.use_tpu)

 estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)

if __name__ == "__main__":
  tf.app.run()
