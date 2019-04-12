# Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems

## Introduction
This code implements the paper titled "Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems", which accelerates the run-time performance of a deep neural network (DNN) running on microcontroller-grade resource-constrained embedded systems. Neuro.ZERO is a novel co-processor architecture consisting of a main microcontroller unit (MCU) that executes scaled-down versions of deep neural network inference tasks, and an accelerator microcontroller unit that is powered by harvested energy. This code implements the four modes of acceleration, i.e., extended inference, expedited inference, ensembled inference, and latent training for Texas Instruments's [MSP430FRXXXX microcontroller](http://www.ti.com/product/MSP430FR5994).

## Install and Setup 
It requires to install Python, TensorFlow (offline training) and Code Composer Studio (binary generation for MSP430FRXXXX). The path evironment also needs to be set for the execution of eclipse (CCS).

1. Install [Python](https://www.python.org/downloads/).

2. Install [Tensorflow](https://www.tensorflow.org/).

3. Install [Code Composer Studio (CCS)](http://www.ti.com/tool/CCSTUDIO).

4. Clone the Neuro.ZERO repository
```sh
$ git clone https://github.com/learning1234embed/Neuro.ZERO.git
Cloning into 'Neuro.ZERO'...
remote: Enumerating objects: 456, done.
remote: Counting objects: 100% (456/456), done.
remote: Compressing objects: 100% (256/256), done.
remote: Total 456 (delta 203), reused 393 (delta 166), pack-reused 0
Receiving objects: 100% (456/456), 534.42 KiB | 4.49 MiB/s, done.
Resolving deltas: 100% (203/203), done.
```

5. Set PATH for eclipse
```sh
export PATH=$PATH:/home/ti/ccsv8/eclipse/
```

## Extended Inference
The following shows an example of generating extended inference with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

1. Create a baseline network to be extended (accelerated) with TensorFlow.
```sh
$ python NeuroZERO.py --mode=b --layers=28*28*1,3*3*1*2,3*3*2*4,3*3*4*8,64,128,10
[] Create a new NeuroZERO
[b] constructing a baseline network
[b] layers: 28*28*1,3*3*1*2,3*3*2*4,3*3*4*8,64,128,10
constructNetwork 1: [[28, 28, 1], [3, 3, 1, 2], [3, 3, 2, 4], [3, 3, 4, 8], [64], [128], [10]]
layer_type: ['input', 'conv', 'max_pool', 'conv', 'max_pool', 'conv', 'max_pool', 'hidden', 'hidden', 'output']
num_of_neuron_per_layer: [[28, 28, 1], [13, 13, 2], [5, 5, 4], [1, 1, 8], [64], [128], [10]]
num_of_weight_per_layer: [18, 72, 288, 512, 8192, 1280]
num_of_bias_per_layer: [2, 4, 8, 64, 128, 10]
layers [[28, 28, 1], [3, 3, 1, 2], [3, 3, 2, 4], [3, 3, 4, 8], [64], [128], [10]]
Tensor("neuron_0:0", shape=(?, 28, 28, 1), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_0:0' shape=(3, 3, 1, 2) dtype=float32_ref>, 'biases': <tf.Variable 'bias_0:0' shape=(2,) dtype=float32_ref>}
new_neuron Tensor("neuron_1:0", shape=(?, 13, 13, 2), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_1:0' shape=(3, 3, 2, 4) dtype=float32_ref>, 'biases': <tf.Variable 'bias_1:0' shape=(4,) dtype=float32_ref>}
new_neuron Tensor("neuron_2:0", shape=(?, 5, 5, 4), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_2:0' shape=(3, 3, 4, 8) dtype=float32_ref>, 'biases': <tf.Variable 'bias_2:0' shape=(8,) dtype=float32_ref>}
new_neuron Tensor("neuron_3:0", shape=(?, 1, 1, 8), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_3:0' shape=(8, 64) dtype=float32_ref>, 'biases': <tf.Variable 'bias_3:0' shape=(64,) dtype=float32_ref>}
new_neuron Tensor("neuron_4:0", shape=(?, 64), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_4:0' shape=(64, 128) dtype=float32_ref>, 'biases': <tf.Variable 'bias_4:0' shape=(128,) dtype=float32_ref>}
new_neuron Tensor("neuron_5:0", shape=(?, 128), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_5:0' shape=(128, 10) dtype=float32_ref>, 'biases': <tf.Variable 'bias_5:0' shape=(10,) dtype=float32_ref>}
new_neuron Tensor("neuron_6:0", shape=(?, 10), dtype=float32)
[] Save NeuroZERO

```
**--mode**: Which command the Neuro.ZERO performs. The example creates a baseline (--mode=b) network.   
**--layers**: The layers and architecture of the network to be created. The example creates a network with total seven layers, i.e., 28\*28\*1 (input), 3\*3\*1\*2 (Conv1), 3\*3\*2\*4 (Conv2), 3\*3\*4\*8 (Conv3), 64 (Fully-connected 1), 128 (Fully-connected 2), 10 (output).


2. Train the newly-created baseline network with TensorFlow.
```sh
$ python NeuroZERO.py --mode=t --network=baseline --data=mnist_data
[t] train
[t] network: baseline
[t] data: mnist_data train/test.size: (55000, 784) (10000, 784)
train
step 0, training accuracy: 0.080000
step 0, validation accuracy: 0.103200
step 100, training accuracy: 0.190000
step 100, validation accuracy: 0.159900
step 200, training accuracy: 0.650000
step 200, validation accuracy: 0.670500
step 300, training accuracy: 0.680000
...
step 4800, training accuracy: 0.900000
step 4800, validation accuracy: 0.928200
step 4900, training accuracy: 0.950000
step 4900, validation accuracy: 0.923500
step 4999, training accuracy: 0.940000
step 4999, validation accuracy: 0.930200
took 23018.169 ms
[] Save NeuroZERO
```
**--mode=t**: Which command the Neuro.ZERO performs. The example trains a network (--mode=t).   
**--network**: Which network to train. The example trains the baseline (baseline) network.
**--data**: The train data. The example uses MNIST data for training.

Create an extended network based on the baseline network
python NeuroZERO.py --mode=ext

Train the extended network
python NeuroZERO.py --mode=t --network=extended --data=mnist_data

Export the network architecture and parameters of the baseline network
python NeuroZERO.py --mode=e --network=baseline

Export the network architecture and parameters of the extended network
python NeuroZERO.py --mode=e --network=extended

Generate (build) the main MCU and accelerator binary
python generate_binary.py --mode=ext
