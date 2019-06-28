import tensorflow as tf
from distutils.dir_util import copy_tree
from chiron_input import read_raw_data_sets
from cnn import getcnnfeature
from cnn import getcnnlogit
from rnn import rnn_layers_one_direction
import time,os
import numpy as np

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
    train_op = adam.minimize(ctc_loss, name="train_op", global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=ctc_loss, train_op=train_op)

def input_fn():
    train_ds = read_raw_data_sets(FLAGS.data_dir,FLAGS.cache_dir,FLAGS.sequence_len,k_mer = FLAGS.k_mer)
    batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size)
    x={"x": batch_x }
    y={ "y": batch_y ,"seq_len": seq_len }
    return x, y 

def run(args):
    global FLAGS
    FLAGS=args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = tf.estimator.Estimator( model_fn=model_fn)
    estimator.train(input_fn=input_fn, max_steps=FLAGS.max_steps)


if __name__ == "__main__":
    class Flags():
     def __init__(self):
        self.data_dir = '/uufs/chpc.utah.edu/common/home/u1142888/Chiron-0.3/new_out'
        self.cache_dir = '/uufs/chpc.utah.edu/common/home/u1142888/Chiron-0.3/output_cache'
        self.log_dir = '/uufs/chpc.utah.edu/common/home/u1142888/Chiron-0.3'
        self.sequence_len = 300
        self.batch_size = 400
        self.step_rate = 1e-3 
        self.max_steps = 1000
        self.k_mer = 1
        self.model_name = 'res50'
        self.retrain =False
    flags=Flags()
    run(flags)
        
        
