# MLgebra

A machine learning tool for Python, in Python.

## New code structure

I have prepared a new code structure for this library.
It is much more readable now. Layout is spreaded to 
different files which are imported as a chain. "Node" name
is now "Neuron". Layers are generated on a base class. I have
included only the Dense layers for now. Model imports layers
and neurons, and manages the usage of them.

One can see 2 main problems with this new code. 

The first one is the lack of so-called "model compiler". Compiling a model
on the given choices (Activation function, error function, etc.)
will result in a much more faster model. This is mainly because,
here at each iteration and pass through, the code checks parameters
in if-elif-else chains. This is not needed if we decide at compile
time. However, it is hard to program this concept purely in Python
as the given objective here.

The second problem, this code is around 10 times slower than the
main branch. The code on main branch is not readable, it is almost
impossible to understand. I don't even know what i did there. But
it is fast (in Python standards). My experience with this unreadable
code was training hundreds of models in a given day. It was possible
to train literally hundreds of models in a day. It was fast enough to
do that. And considering you can train multiple models at once, I was
doing around 1000 models per day at the fastest pace. 

It is impossible to do that here. I have profiled the code, most of
the time is spent at matrix multiplication. And for some reason, 
at the forward pass, this multiplication is 15 to 20 times slower than
vanilla Vectorgebra. I am yet unable to find the reason and therefore
the solution. I will create an issue for that, if you are able to help,
you are welcome!

## Project details

https://pypi.org/project/mlgebra/

[Github](https://github.com/ahmeterdem1/ml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

_pip install mlgebra_

This project is a continuation of [this](https://github.com/ahmeterdem1/MLgebra) 
repository. Which is also a continuation of _vectorgebra_ library. This particular
library therefore requires vectorgebra to be installed. Currently, the development
process continues and the code is probably unstable. 

This is a basic tool of machine learning that is created based on vectorgebra. All
internal operations are imported from vectorgebra. Specific algorithms for machine
learning are created based upon them. 

There are 2 main classes; Node, Model.

Node represents neurons in the neural network. They hold the weights connecting it
to the previous layer in the model as a vector.

## Model

Model class has all functionality to train one. Initialization starts with giving it 
a name, choosing whether to use decimal library and to use multiprocessing. Multiprocessing
should only be used when dealing with large models and large data batch sizes. Otherwise,
linear program flow is better. 

Model name must be unique. Weights and biases of the model are saved to a file according to
this given name. Initialization will raise an error if a weight file with the same name is 
found in the same directory.

Printing the model object will print out details about the model.

### _Model_.addDense(amount)

The only layer type currently is "dense". This method adds a layer of "amount" neurons to the
model.

### _Model_.saveModel()

Saves the weight file of the model.

### _Model_.readWeightFile(path)

Loads a weight file specified via "path" to the model. This operation must be done without any 
bias or weight finalization. However, layers must be configured before using it.

### _Model_.readMNIST(path1, path2, path3, path4)

Reads training/test data/label of MNIST database and returns it in order. Paths are in order to 
training data, training labels, test data, test labels. Returned objects are lists of vectors.
They can be directly used for .singleTrain(), oru you can group them and use them in parallel.

### _Model_.finalize(generation="flat", a=-2, b=2)

This method "finalizes" the model. Therefore, it should be called at the end of every other thing.
This method initializes all the weights to layers according to the given "generation" method. This
can be _uxavier_, _nxavier_,  _he_, _naive_ and _flat_. _uxavier_ is uniform Xavier initalization.
_nxavier_ is normal Xavier initialization. _he_ is He initialization. _naive_ is initialization from
normal distribution. If this method is choosen, "a" is the standard deviation of the curve. _flat_ is
just random float generation from the range [a, b).

### _Model_.includeBias(generation="flat", a=-2, b=2)

This method includes bias vectors to the model. It can be omitted if you don't want to use any bias.
Of course this is not a very good idea but being able to omit it adds experimentability to the library.

_flat_, _zero_ and _constant_ are possible generation methods. _flat_ is generation from random floats in
range [a, b). _zero_ initializes all biases to 0, and _constant_ initializes all them to a constant value
given by "a".

### _Model_.configureMethods(**kwargs)

This method configures all other parameters of your model. Activation function, input normalization, operations
to the output logits and parameters to all related functions are given here. 

All keys: _input_, _output_, _activator_, _cutoff_, _leak_

All possible choices: _minmax_, _softmax_, _sigmoid_, _relu_, cutoff and leak accepts numerical values

These possibilities will be more diverse in the upcoming versions. Currently, these are all.

### _Model_.updateMatrices()

This is a helper function called from training methods. _self.last_output_ must be non-empty to use this.
You can use this method if you want to use different training approaches with your own function definitions.
Currently only optimizer method that is being used by training functions is stochastic gradient descent.

### _Model_.singleTrain(data, label, learing_rate)

This function trains the model by the given singular data. "data" must either be a Vectorgebra.Vector or
Vectorgebra.Matrix. "label" must be a Vectorgebra.Vector. You can choose "learning_rate" as you wish.

### _Model_.train(dataset, labelset, learning_rate)

This method trains the model by data batches. "dataset" must be a list of all Vectorgebra.Vector or 
Vectorgebra.Matrix. "labelset" must be a list of all Vectorgebra.Vector. Length of these two lists
must be equal. If, during initialization, multiprocessing was chosen to be True, this method applies
parallel programming. 

### _Model_.produce(data)

This method produces a result by the model. "data" must be either a Vectorgebra.Vector or Vectorgebra.Matrix.
Result is a Vectorgebra.Vector object which is the result values from the output layer logits.

<hr>

## Exceptions

### FileStructureError

This exception is raised when files extension is wrong.

### ConfigError

This function is raised when any problem with the model configuration occurs at any point of the structuring.


