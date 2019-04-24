from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train_set():
	return mnist.train.images, mnist.train.labels
	
def validation_set():
	return mnist.validation.images, mnist.validation.labels
	
def test_set():
	return mnist.test.images, mnist.test.labels
