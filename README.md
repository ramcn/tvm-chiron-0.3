# TVM Chiron

## Mapping A basecaller for Oxford Nanopore Technologies' sequencers to TVM
Using a deep learning CNN+RNN+CTC structure to establish end-to-end basecalling for the nanopore sequencer.
Built with **TensorFlow** and python 2.7.

If you found Chiron useful, please consider to cite:
> Teng, H., et al. (2017). Chiron: Translating nanopore raw signal directly into nucleotide sequence using deep learning. [bioRxiv 179531] (https://www.biorxiv.org/content/early/2017/09/12/179531)

---
Python 2.7 venv environment for training and Python 3.6 venv for openvino inference and tensorflow 1.10.1 installed in both environment

## Test
To run the training and generate the frozen model run the below commands
```
conda activate tf2.7
python chiron/chiron_rcnn_train.py
conda deactivate
conda activate tf3.6
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_model.pb
python test_ov.py -m frozen_model.xml


