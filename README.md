# Neuro.ZERO
Neuro.ZERO: A Zero-Energy Neural Network Accelerator for Embedded Sensing and Inference Systems

Install Phthon, Tensorflow, CCS

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
