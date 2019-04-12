# Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems

## Introduction
This code implements the paper titled "Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems", which accelerates the run-time performance of a deep neural network (DNN) running on microcontroller-grade resource-constrained embedded systems. Neuro.ZERO is a novel co-processor architecture consisting of a main microcontroller unit (MCU) that executes scaled-down versions of deep neural network inference tasks, and an accelerator microcontroller unit that is powered by harvested energy. This code implements the four modes of acceleration, i.e., extended inference, expedited inference, ensembled inference, and latent training for Texas Instruments's [MSP430FR5994 microcontroller](http://www.ti.com/product/MSP430FR5994).

## Install and Setup 
It requires to install Python, TensorFlow and ode Composer Studio. The path evironment also needs to be set for the execution of eclipse (CCS).

1. Install [Python](https://www.python.org/downloads/).

2. Install [Tensorflow](https://www.tensorflow.org/).

3. Install [Code Composer Studio (CCS)](http://www.ti.com/tool/CCSTUDIO).

4. Clone the Neuro.ZERO repository
```sh
git clone https://github.com/learning1234embed/Neuro.ZERO.git
```

5. Set PATH for eclipse
```sh
export PATH=$PATH:/home/ti/ccsv8/eclipse/
```

## Extended Inference

Create a baseline network
python NeuroZERO.py --mode=b --layers=28*28*1,3*3*1*2,3*3*2*4,3*3*4*8,64,128,10

Train the baseline network
python NeuroZERO.py --mode=t --network=baseline --data=mnist_data

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
