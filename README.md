# Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems

## Introduction
This repository implements the paper titled **"Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems"**, which accelerates the run-time performance of a deep neural network (DNN) running on microcontroller-grade resource-constrained embedded systems. Neuro.ZERO is a novel co-processor architecture consisting of a main microcontroller unit (MCU) that executes scaled-down versions of deep neural network inference tasks, and an accelerator microcontroller unit that is powered by harvested energy.

It implements Neuro.ZERO's 1) *extended inference* that increases the classification accuracy, and 2) *latent training* that performs on-device online training of a DNN with the adaptive-scal fixed-point and skip-out training algorithm, by automatically generating an executable binary which run on the Texas Instruments's [MSP430FRXXXX microcontroller](http://www.ti.com/product/MSP430FR5994).

&nbsp;
## Software Install and Setup 
Neuro.ZERO requires Python, TensorFlow (for training) and Code Composer Studio (for MSP430FRXXXX binary generation). The path environment also needs to be set for the execution of eclipse (CCS).

Step 1. Install [Python](https://www.python.org/downloads/).

Step 2. Install [Tensorflow](https://www.tensorflow.org/).

Step 3. Install [Code Composer Studio (CCS)](http://www.ti.com/tool/CCSTUDIO) and set PATH for eclipse execution.
```sh
export PATH=$PATH:/home/ti/ccsv8/eclipse/
```
Step 4. Clone the Neuro.ZERO repository.
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

&nbsp;
## 1) Extended Inference (Step by Step Guide with MNIST)
The following shows an example of generating extended inference with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

Step 1. Create a baseline network to be extended (accelerated) with TensorFlow.
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
**--mode**: Which command the Neuro.ZERO performs. The example creates a baseline network  (--mode=b).   
**--layers**: The layers and architecture of the network to be created. The example creates a network with total seven layers, i.e., 28\*28\*1 (input), 3\*3\*1\*2 (Conv1), 3\*3\*2\*4 (Conv2), 3\*3\*4\*8 (Conv3), 64 (Fully-connected 1), 128 (Fully-connected 2), 10 (output).


Step 2. Train the newly-created baseline network with TensorFlow.
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
...
step 4800, training accuracy: 0.900000
step 4800, validation accuracy: 0.920200
step 4900, training accuracy: 0.950000
step 4900, validation accuracy: 0.923500
step 4999, training accuracy: 0.940000
step 4999, validation accuracy: 0.925200
took 23018.169 ms
[] Save NeuroZERO
```
**--mode**: Which command the Neuro.ZERO performs. The example trains a network (--mode=t).   
**--network**: Which network to train. The example trains the baseline network (--network=baseline).  
**--data**: The train data. The example uses MNIST data for training (--data=mnist_data).


Step 3. Create an extended network by expanding the baseline network, which is expected to provide better performance (higher classification accuracy). Its architecture and layers formation are automatically determined based on the baseline network.
```sh
$ python NeuroZERO.py --mode=ext
[] Load NeuroZERO
[ext] constructing a extended network
constructNetworkExtended 2: [[28, 28, 1], [3, 3, 1, 2], [3, 3, 2, 4], [3, 3, 4, 8], [32], [64], [10]]
base_network: <__main__.Network instance at 0x7f3267b97050>
layer_type: ['input', 'conv', 'max_pool', 'conv', 'max_pool', 'conv', 'max_pool', 'hidden', 'hidden', 'output']
num_of_neuron_per_layer: [[28, 28, 1], [13, 13, 2], [5, 5, 4], [1, 1, 8], [32], [64], [10]]
num_of_weight_per_layer: [18, 72, 288, 256, 2048, 640]
num_of_bias_per_layer: [2, 4, 8, 32, 64, 10]
layers [[28, 28, 1], [3, 3, 1, 2], [3, 3, 2, 4], [3, 3, 4, 8], [64], [128], [10]]
Tensor("neuron_0:0", shape=(?, 28, 28, 1), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_0_base:0' shape=(3, 3, 1, 2) dtype=float32_ref>, 'biases': <tf.Variable 'bias_0_base:0' shape=(2,) dtype=float32_ref>}
new_neuron Tensor("neuron_1_base:0", shape=(?, 13, 13, 2), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_1_base:0' shape=(3, 3, 2, 4) dtype=float32_ref>, 'biases': <tf.Variable 'bias_1_base:0' shape=(4,) dtype=float32_ref>}
new_neuron Tensor("neuron_2_base:0", shape=(?, 5, 5, 4), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_2_base:0' shape=(3, 3, 4, 8) dtype=float32_ref>, 'biases': <tf.Variable 'bias_2_base:0' shape=(8,) dtype=float32_ref>}
new_neuron Tensor("neuron_3_base:0", shape=(?, 1, 1, 8), dtype=float32)
Tensor("neuron_0:0", shape=(?, 28, 28, 1), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_0:0' shape=(3, 3, 1, 2) dtype=float32_ref>, 'biases': <tf.Variable 'bias_0:0' shape=(2,) dtype=float32_ref>}
new_neuron Tensor("neuron_1:0", shape=(?, 13, 13, 2), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_1:0' shape=(3, 3, 2, 4) dtype=float32_ref>, 'biases': <tf.Variable 'bias_1:0' shape=(4,) dtype=float32_ref>}
new_neuron Tensor("neuron_2:0", shape=(?, 5, 5, 4), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_2:0' shape=(3, 3, 4, 8) dtype=float32_ref>, 'biases': <tf.Variable 'bias_2:0' shape=(8,) dtype=float32_ref>}
new_neuron Tensor("neuron_3:0", shape=(?, 1, 1, 8), dtype=float32)
fc_parameter {'weights': <tf.Tensor 'weight_3:0' shape=(16, 96) dtype=float32>, 'biases': <tf.Variable 'bias_3:0' shape=(96,) dtype=float32_ref>}
new_neuron Tensor("neuron_4:0", shape=(?, 96), dtype=float32)
fc_parameter {'weights': <tf.Tensor 'weight_4:0' shape=(96, 192) dtype=float32>, 'biases': <tf.Variable 'bias_4:0' shape=(192,) dtype=float32_ref>}
new_neuron Tensor("neuron_5:0", shape=(?, 192), dtype=float32)
fc_parameter {'weights': <tf.Tensor 'weight_5:0' shape=(192, 10) dtype=float32>, 'biases': <tf.Variable 'bias_5:0' shape=(10,) dtype=float32_ref>}
new_neuron Tensor("neuron_6:0", shape=(?, 10), dtype=float32)
[] Save NeuroZERO
```
**--mode**: Which command the Neuro.ZERO performs. The example creates an extended (ext) network (--mode=ext).   


Step 4. Train the extended network.
```sh
$ python NeuroZERO.py --mode=t --network=extended --data=mnist_data
[t] train
[t] network: extended
[t] data: mnist_data train/test.size: (55000, 784) (10000, 784)
train
step 0, training accuracy: 0.780000
step 0, validation accuracy: 0.827800
step 100, training accuracy: 0.900000
step 100, validation accuracy: 0.930900
step 200, training accuracy: 0.940000
step 200, validation accuracy: 0.930200
...
step 4800, training accuracy: 0.980000
step 4800, validation accuracy: 0.972700
step 4900, training accuracy: 0.960000
step 4900, validation accuracy: 0.970300
step 4999, training accuracy: 0.980000
step 4999, validation accuracy: 0.972300
```
**--mode**: Which command the Neuro.ZERO performs. The example trains a network (--mode=t).   
**--network**: Which network to train. The example trains the extended network (--network=extended).  
**--data**: The train data. The example uses MNIST data for training (--data=mnist_data).


Step 5. Export the network architecture and parameters of the baseline network for MSP430FRXXXX binary generation.
```sh
$ python NeuroZERO.py --mode=e --network=baseline
[] Load NeuroZERO
[e] export weights and biases of a network to a file
weight_len 10362
bias_len 216
weight_len 10362
bias_len 216
[] Save NeuroZERO
```
**--mode**: Which command the Neuro.ZERO performs. The examples exports (--mode=e) the architecture and parameters of the network.   
**--network**: Which network to export. The example exports the baseline network (--network=baseline).  


Step 6. Export the network architecture and parameters of the extended network for MSP430FRXXXX binary generation.
```sh
$ python NeuroZERO.py --mode=e --network=extended
[] Load NeuroZERO
[e] export weights and biases of a network to a file
weight_len 22266
bias_len 312
weight_len 22266
bias_len 312
[] Save NeuroZERO
```
**--mode**: Which command the Neuro.ZERO performs. The examples exports (--mode=e) the architecture and parameters of the network.   
**--network**: Which network to export. The example exports the extended network (--network=extended). 


Step 7. Generate (compile) the main MCU and accelerator binary (MSP430FRXXXX). The code and output binary for the main MCU and the accelerator are located at the 'extended_MAIN/' and 'extended_ACC/' folders, respectively. The code for each MCU can be edited by a user and compiled mutilple times as needed.
```sh
$ python generate_binary.py --mode=ext
Neuro.ZERO/extended_MAIN created
baseline_param.h generated and copied
extended_param.h generated and copied
start compiling main MCU
eclipse -noSplash -data "./" -application com.ti.ccstudio.apps.projectBuild -ccs.configuration Debug -ccs.autoImport -ccs.projects extended_MAIN

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CCS headless build starting... [Fri Apr 12 17:18:21 EDT 2019] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

================================================================================
Pre processing...


================================================================================
Building...


**** Build of configuration Debug for project extended_MAIN ****

/home/XXX/ti/ccsv8/utils/bin/gmake -k -j 12 all -O 
 
Building file: "../DSPLib/source/filter/msp_biquad_cascade_df2_q15.c"
Invoking: MSP430 Compiler

...

Finished building target: "extended_ACC.out"
 

**** Build Finished ****


================================================================================
CCS headless build complete! 0 out of 1 projects have errors.

```

&nbsp;
## 2) Latent Training
The following python script generates an executable binary for MSP430FRXXXX which performs on-device online training on the accelerator. The trainining is performed with the proposed *Adaptive-Scale Fixed-Point (ASFP)* arithmetic and *Skip-Out training algorithm* as described in the paper. The standard momentum gradient-update method and ReLU are used for online training.
```sh
$ python generate_binary.py --mode=latent
Neuro.ZERO/latent_train_ACC created
start compiling accelerator
eclipse -noSplash -data "./" -application com.ti.ccstudio.apps.projectBuild -ccs.configuration Debug -ccs.autoImport -ccs.projects latent_train_ACC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CCS headless build starting... [Tue Apr 23 19:46:28 EDT 2019] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

================================================================================
Pre processing...


================================================================================
Building...


**** Build of configuration Debug for project latent_train_ACC **** 

/home/XXX/ti/ccsv8/utils/bin/gmake -k -j 12 all -O 
 
Building file: "../DSPLib/source/filter/msp_biquad_cascade_df2_ext_q15.c"
Invoking: MSP430 Compiler

...

Finished building target: "latent_train_ACC.out"
 

**** Build Finished ****


================================================================================
CCS headless build complete! 0 out of 1 projects have errors.
```

When executing the generated latent training binary on the accelerator with CCS, the following log will come. It trains a fully-connected network with three layers (2x64x2) as an example. The architecture and layer of the network can be changed by editing the variables in the code.
```sh
weight_input
weight_output

iteration 00000
[0] 0.90000 0.10000 : 0.50062 0.49938 : 1.00000 0.00000 :loss = 0.691911876
[1] 0.10000 0.90000 : 0.49958 0.50042 : 0.00000 1.00000 :loss = 0.692308426
total_loss = 1.384220362

back_propagate[0] skipout = 0.018585773
back_propagate[1] skipout = 0.190466017

...

iteration 00118
[0] 0.90000 0.10000 : 0.62980 0.37020 : 1.00000 0.00000 :loss = 0.462358147
[1] 0.10000 0.90000 : 0.45789 0.54211 : 0.00000 1.00000 :loss = 0.612280250
total_loss = 1.074638367

back_propagate[0] skipout = 0.547532558
back_propagate[1] skipout = 0.196020380

...

iteration 00441
[0] 0.90000 0.10000 : 0.98813 0.01187 : 1.00000 0.00000 :loss = 0.011939470
[1] 0.10000 0.90000 : 0.01556 0.98444 : 0.00000 1.00000 :loss = 0.015686356
total_loss = 0.027625825

back_propagate[0] skipout = 0.534562230
back_propagate[1] skipout = 0.089632861
```
