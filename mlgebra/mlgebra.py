from model import *
from time import time
logger.setLevel(logging.WARN)

mymodel = Model("MyModel")

mymodel.structure([
    Flatten(784),
    Dense(784, 784, "minmax"),
    Dense(784, 32, "relu"),
    Dense(32, 16, "relu"),
    Dense(16, 10, "softmax")
])

#mymodel.loadModel("./MyModel.model")

mymodel.describe()
m = Matrix.randMint(28, 28, 0, 256)
v = Vector(1,0,0,0,0,0,0,0,0,0)
random_x = [m] * 10
random_y = [v] * 10
begin = time()
#for k in range(1000):
#    model.produce(m)
for k in range(100):
    mymodel.train(random_x, random_y)
end = time()
print(f"{(end - begin)/100} seconds on average.")

#mymodel.compile("/Users/ahmeterdem/Desktop")
"""
mnist = "../../vtest/mnist/"
# ../../vtest/mnist/

x_train, y_train, x_test, y_test = readMnist(mnist + "train-images.idx3-ubyte",
                                             mnist + "train-labels.idx1-ubyte",
                                             mnist + "t10k-images.idx3-ubyte",
                                             mnist + "t10k-labels.idx1-ubyte")



mymodel.train(x_train, y_train, lr=0.0001)
#print(f"Calculated exact error: {mymodel.evaluate(x_test, y_test)}")
mymodel.saveModel(".")

total = 0
for k in range(100):
    v = mymodel.produce(x_test[k])
    print(v)
    if v.values.index(maximum(v)) == y_test[k].values.index(1):
        total += 1

print(f"{(total / 100):.2f} accuracy at first 100 images.")"""


