import mnist_loader
import network

print('starting process')

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#net.SGD parameters:
#number of image pixels; 28x28=784
#number of neurons in hidden layer
#output neurons [0,1,2,3,4,5,6,7,8,9]
net = network.Network([784, 30, 10])

#net.SGD parameters:
#training_data
#number of epoches to run
#batch size
#learning rate, eta
#test_data
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)

