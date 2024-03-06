from model import *

mymodel = Model("MyModel")

mymodel.structure([
    Dense(784, 784, "minmax"),
    Dense(784, 32, "relu"),
    Dense(32, 16, "relu"),
    Dense(16, 10, "softmax")
])

mymodel.describe()

mnist = "DIR_TO_MNIST/"

x_train, y_train, x_test, y_test = readMnist(mnist + "train-images.idx3-ubyte",
                                             mnist + "train-labels.idx1-ubyte",
                                             mnist + "t10k-images.idx3-ubyte",
                                             mnist + "t10k-labels.idx1-ubyte")

mymodel.train(x_train, y_train)
print(f"Calculated exact error: {mymodel.evaluate(x_test, y_test)}")
mymodel.saveModel(".")
