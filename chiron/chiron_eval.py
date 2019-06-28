import argparse,os,time,sys
import numpy as np
import tensorflow as tf
from chiron_input import read_data_for_eval
from utils.easy_assembler import simple_assembly
from utils.easy_assembler import simple_assembly_qs
#from utils.easy_assembler import section_decoding
from cnn import getcnnfeature
from cnn import getcnnlogit
from rnn import rnn_layers
from utils.unix_time import unix_time

import tvm
from tvm import relay
import tvm.relay.testing.tf as tf_testing

def inference(x,seq_length,training):
    cnn_feature = getcnnfeature(x,training = training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = FLAGS.segment_len/feashape[1]
    #logits = rnn_layers(cnn_feature,seq_length/ratio,training,class_n = 5 )
#   logits = rnn_layers_one_direction(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 )
    logits = getcnnlogit(cnn_feature)
    return logits,ratio

def path_prob(logits):
    top2_logits = tf.nn.top_k(logits,k=2)[0]
    logits_diff = tf.slice(top2_logits,[0,0,0],[FLAGS.batch_size,FLAGS.segment_len,1])-tf.slice(top2_logits,[0,0,1],[FLAGS.batch_size,FLAGS.segment_len,1])
    prob_logits = tf.reduce_mean(logits_diff,axis = -2)
    return prob_logits

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="prefix")
    return graph

def evaluation():
    x = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape = [FLAGS.batch_size])
    training = tf.placeholder(tf.bool)

    logits,_ = inference(x,seq_length,training = training)
    predict = tf.nn.ctc_greedy_decoder(tf.transpose(logits,perm=[1,0,2]),seq_length,merge_repeated = True)
    prob = path_prob(logits)

    target = 'llvm'
    target_host = 'llvm'
    layout = None
    ctx = tvm.cpu(0)

    #graph = load_graph("frozen_model.pb")
    graph = load_graph("optimized_model.pb")
    #for op in graph.get_operations():
    #    print(op.name)
    x = graph.get_tensor_by_name('prefix/Placeholder_1:0')
    y = graph.get_tensor_by_name('prefix/cnnlogits_rs:0')


    #with tf.Session() as sess:
    with tf.Session(graph=graph) as sess:
     #saver = tf.train.Saver()
     print(FLAGS.model) 
     #saver.restore(sess,tf.train.latest_checkpoint(FLAGS.model))

     file_list = os.listdir(FLAGS.input)
     file_dir = FLAGS.input

     shape_dict = None
     sym, params = relay.frontend.from_tensorflow(sess.graph_def, layout=layout, shape=shape_dict)
     print ("Tensorflow protobuf imported to relay frontend.")

     for name in file_list:
         start_time = time.time()
         if not name.endswith('.signal'):
             continue
         file_pre = os.path.splitext(name)[0]
         input_path = os.path.join(file_dir,name)
         eval_data = read_data_for_eval(input_path,FLAGS.start,seg_length = FLAGS.segment_len,step = FLAGS.jump)
         reads_n = eval_data.reads_n
         reading_time=time.time()-start_time
         reads = list()
         for i in range(0,reads_n,FLAGS.batch_size):
             batch_x,seq_len,_ = eval_data.next_batch(FLAGS.batch_size,shuffle = False)
             batch_x=np.pad(batch_x,((0,FLAGS.batch_size-len(batch_x)),(0,0)),mode='constant')
             seq_len=np.pad(seq_len,((0,FLAGS.batch_size-len(seq_len))),mode='constant')
             #feed_dict = {x:batch_x,seq_length:seq_len,training:False}
             #logits_val = sess.run(logits,feed_dict = feed_dict)
             feed_dict = {x:batch_x}
             #logits_val = sess.run(y,feed_dict = feed_dict)
             #print(logits_val)

def run(args):
    global FLAGS
    FLAGS=args
    time_dict=unix_time(evaluation)
    print(FLAGS.output)
    print('Real time:%5.3f Systime:%5.3f Usertime:%5.3f'%(time_dict['real'],time_dict['sys'],time_dict['user']))
    meta_folder = os.path.join(FLAGS.output,'meta')
    if os.path.isdir(FLAGS.input):
        file_pre='all'
    else:
        file_pre = os.path.splitext(os.path.basename(FLAGS.input))[0]
    path_meta=os.path.join(meta_folder,file_pre+'.meta')
if __name__=="__main__":
    parser=argparse.ArgumentParser(prog='chiron',description='A deep neural network basecaller.')
    parser.add_argument('-i','--input',default='example_data/output/raw', help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o','--output',default='example_data/output', help = "Output Folder name")
    parser.add_argument('-m','--model', default = 'model/DNA_default',help = "model folder")
    parser.add_argument('-s','--start',type=int,default = 0,help = "Start index of the signal file.")
    parser.add_argument('-b','--batch_size',type = int,default = 1100,help="Batch size for rune processing speed and give a slightly better accuracy but require larger RAM load")
    parser.add_argument('-l','--segment_len',type = int,default = 300, help="Segment length to be divided into.")
    parser.add_argument('-j','--jump',type = int,default = 30,help = "Step size for segment")
    parser.add_argument('-t','--threads',type = int,default = 0,help = "Threads number")
    parser.add_argument('-e','--extension',default = 'fastq',help = "Output file extension.")
    parser.add_argument('--beam',type = int,default = 0, help = "beam width give better decoding result but require longer decoding time.")
    args=parser.parse_args(sys.argv[1:])
    run(args)

