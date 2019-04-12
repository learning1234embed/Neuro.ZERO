import struct
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train_set():
	return mnist.train.images, mnist.train.labels
	
def validation_set():
	return mnist.validation.images, mnist.validation.labels
	
def test_set():
	return mnist.test.images, mnist.test.labels

def write_data():
	f = open('mnist_test', 'w')
	print mnist.test.images.shape
	data_len = mnist.test.images.shape[0]
	single_data_size = mnist.test.images.shape[1]
	print 'data_len', data_len
	print 'single_data_size', single_data_size

	f.write(struct.pack('i', data_len))
	f.write(struct.pack('i', single_data_size))

	for image in mnist.test.images:
		#print len(image)
		for i in range(len(image)):
			f.write(struct.pack('f', image[i]))

	f.close()

def write_label():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
	f = open('mnist_label', 'w')
	data_len = mnist.test.labels.shape[0]
	#single_data_size = mnist.test.labels.shape[1]
	single_data_size = 1
	print 'data_len', data_len
	print 'single_data_size', single_data_size

	f.write(struct.pack('i', data_len))
	f.write(struct.pack('i', single_data_size))

	for label in mnist.test.labels:
		f.write(struct.pack('b', label))
	f.close()


def main():
	"""
	write_data()
	write_label()
	"""
	return

if __name__ == '__main__':
	main()
