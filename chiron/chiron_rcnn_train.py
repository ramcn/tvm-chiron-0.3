import tensorflow as tf
from distutils.dir_util import copy_tree
from chiron_input import read_raw_data_sets
import numpy as np
from cnn import getcnnfeature
from cnn import getcnnlogit
#from rnn import rnn_layers
from rnn import rnn_layers_one_direction
import time,os
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib

def inference(x,seq_length,training):
    cnn_feature = getcnnfeature(x,training = training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = FLAGS.sequence_len/feashape[1]
#    logits = rnn_layers(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 )
#    logits = rnn_layers_one_direction(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 ) 
    logits = getcnnlogit(cnn_feature)
    return logits,ratio

def loss(logits,seq_len,label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(label,logits,seq_len,ctc_merge_repeated = True,time_major = False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss',loss)
    return loss

def train_step(loss,global_step = None):
    opt = tf.train.AdamOptimizer(FLAGS.step_rate).minimize(loss,global_step=global_step)
    return opt

def train():
    training = tf.placeholder(tf.bool)
    global_step=tf.get_variable('global_step',trainable=False,shape=(),dtype = tf.int32,initializer = tf.zeros_initializer())
    x = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape = [FLAGS.batch_size])
    logits, ratio = inference(x,seq_length,training)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    print("Model init finished, begin loading data. \n")

    train_ds = read_raw_data_sets(FLAGS.data_dir,FLAGS.cache_dir,FLAGS.sequence_len,k_mer = FLAGS.k_mer)
    start=time.time()
    for i in range(FLAGS.max_steps):
        batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size)
        batch_x=np.pad(batch_x,((0,FLAGS.batch_size-len(batch_x)),(0,0)),mode='constant')
        seq_len=np.pad(seq_len,((0,FLAGS.batch_size-len(seq_len))),mode='constant')
        feed_dict =  {x:batch_x,seq_length:seq_len,training:False}
        logits_val = sess.run(logits,feed_dict = feed_dict)

    global_step_val = tf.train.global_step(sess,global_step)
    saver.save(sess,FLAGS.log_dir+FLAGS.model_name+'/final.ckpt',global_step=global_step_val)

    saver = tf.train.import_meta_graph('./res50/final.ckpt-0.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess1 = tf.Session()
    saver.restore(sess1, "./res50/final.ckpt-0")
    output_node_names="cnnlogits_rs"
    output_graph_def = graph_util.convert_variables_to_constants(sess1, input_graph_def, output_node_names.split(","))
    output_graph="./frozen_model.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess1.close()
    print("Generated the frozen model")

    inputGraph = tf.GraphDef()
    with tf.gfile.Open('frozen_model.pb', "rb") as f:
      data2read = f.read()
      inputGraph.ParseFromString(data2read)
  
    outputGraph = optimize_for_inference_lib.optimize_for_inference( inputGraph, ["Placeholder_1"], ["cnnlogits_rs"], tf.float32.as_datatype_enum)

    # Save the optimized graph'test.pb'
    f = tf.gfile.FastGFile('optimized_model.pb', "w")
    f.write(outputGraph.SerializeToString()) 
    print ("Completed wrting optimized graph")


def run(args):
    global FLAGS
    FLAGS=args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    train()

if __name__ == "__main__":
    class Flags():
     def __init__(self):
        self.data_dir = '/home/chakenal/pb-p3-chiron-tvm/test_data_raw'
        self.cache_dir = '/home/chakenal/pb-p3-chiron-tvm/output_cache'
        self.log_dir = '/home/chakenal/pb-p3-chiron-tvm'
        self.sequence_len = 300
        self.batch_size = 400
        self.step_rate = 1e-3 
        self.max_steps = 20
        self.k_mer = 1
        self.model_name = 'res50'
        self.retrain = False
    flags=Flags()
    run(flags)
        
        
