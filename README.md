# MLgebra

A machine learning tool for Python, in Python.

## New code structure

I have prepared a new code structure for this library.
It is much more readable now. Layout is spread to 
different files which are imported as a chain. "Node" name
is now "Neuron". Layers are generated on a base class. I have
included only the Dense layers for now. Model imports layers
and neurons, and manages the usage of them.

Problems stated in the last commit is now solved.

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

This library can be used in 2 different ways. These methods can be classified into 2:
Dynamic usage and Static usage.

Dynamic usage, is basically the machine learning library that this code aims to be. 
You can configure your models, load data, and train them. MLgebra also imports 
Vectorgebra, you can use all the tools there too to preprocess your data, etc. This
is what you would expect from a classic library. 

Static usage is inherently different from being a "library". Static usage turns this
library into a framework-like tool. 

Among libraries source files, there are some .spy files. This is short for "source python".
Those files are template mirrors of original libraries class files. If you chose to compile
your model with .compile(_destination_) method, those template files are filled according to your
configurations and a complete mirror of MLgebra library is generated at _destination_ folder. 

Working with this mirror is done by just importing it and using it as if it were MLgebra. 
You can edit the literal source mirror files as you wish. Despite the given flexibility 
to the user with this "Static" method, it is named static because it is still a
lot of lines of code which would take long considerations and trial-and-error cycles to
properly change and improve. 

You can share your static files as you wish and import other Python files etc., which is
one of the main points of this compilation method. Your model in Python, is also compiled
into Python.

### Notes

There is indeed no literal "compilation" like modern AI compilers do. Compilation operation
done here is just a template filling and loop unrolling. A different class is generated for
each layer and its neurons from the base classes Neuron and Layer. Neuron base class is absent
in original MLgebra. Attributes of objects may also differ from original MLgebra as compiled
class files do not require much of the information given in the original code. 

Deeper explanations for the Dynamic usage is given below.

<hr>

## Neuron

This is the basic class for "perceptron". Objects from this class are the building blocks
of layers. Weights connecting the perceptron to the previous layer, are stored in the
perceptron. ".next_weights" attribute is for updating the perceptron. Trained values
are stored in this attribute at each iteration. When ".update()" is called on the layer,
".next_weights" replaces ".weights".

## Layer

Layer is the base class for all types of layers that exists in the library, and that
will exist. Main skeleton for all subclasses are given here. Initializers takes many
arguments; input_shape, units, activation, bias, initialization, decimal, low, high,
bias_low, bias_high, cutoff, leak, template. 

### Dense

Basic dense layer for multilayer perceptron. Currently, the only general purpose
layer type in the library.

#### Initializer arguments

- input_shape: Dimension of the inputted vector to the layer.
- units: Dimension of the layer that will be generated.
- activation: Activation functions name; e.g. "relu", "sigmoid".
- bias: Bias generation method; e.g. "naive", "zero".
- initialization: Weight initialization method; e.g. "xavier", "he".
- decimal: Choice to use Decimal or not: boolean
- low: Parameter for weight generation
- high: Parameter for weight generation
- bias_low: Parameter for bias generation
- bias_high: Parameter for bias generation
- cutoff: Used for activation choices "relu" and "sigmoid". Processed internally.
- leak: Used for activation choice "relu". Processed internally.
- template: Internal boolean. Used by the library when a model is read from file.

#### startForward(input: Vector)

Start the forward pass process. Returns the resulting processed Vector. Input must
be preprocessed. 

#### forward(input: Vector)

Continues the forward pass. Returns the output Vector from the activation function.

#### startBackward(deltas, inputs, lr)

Starts the backpropagation process. "deltas" is the Vector of errors. "inputs" is the
previous inputs to this layer during the forward pass process. "lr" is the learning rate.
Must be a Decimal object if it is set to True in the model. Saves the recalculated weights
in Neurons' ".next_weights" and saves the recalculated biases in self.next_bias. 

#### backward(deltas, inputs, lr)

Continues the backpropagation process. Arguments are the same as .startBackward(). Applies
derivatives according to the activation function of the layer. Other functionalities are
the same as .startBackward().

#### .update()

Replaces weights and biases with the trained ones. Must be called after the backwards pass
ends completely. 

### Flatten

A preprocessing layer that flattens the matrix-tensor input into a Vector. 

#### Initializer arguments

- shape: The dimension of the required output Vector. 

#### startForward(), forward()

Flattens the input tensor into a Vector which is "shape" dimensional.


## Model

This is the class that collects neurons and layers together and governs them as to
train a "model". 

### Initializing arguments

- name: Name of the model. This must be different for each object, as file savings etc. are done according to this name.
- loss: Loss function choice. Only cross entropy loss is included in the library yet. You may leave this be.
- decimal: Main decimal choice of your model is done here. If set True, all other governed objects will have decimal=True.

### loadModel(path)

Loads a saved model file. You have to structure your model accordingly before loading the weights. 
Saved model files only store weights, biases and activation functions. "path" is a complete path
to your model file including its name. This method is currently not usable for compiled model mirror
files. 

### saveModel(path)

Saves your model to the specified directory. "path" is now a directory. File name is generated with your models
name. This method is currently not usable for compiled model mirror files. 

### addLayer(layer: Layer)

Add a layer to your model.

### structure(layers: List)

Give a complete list of layers of your model. A new layer list is generated from the "layers" argument.

### produce(input)

Produce an output with your model. "input" is what your model takes. It might be a Vector, or a Matrix
if you have a Flatten layer. Same for Tensor. Returns the output Vector of your model. 

### evaluate(x, y)

Returns the calculated absolute error of your model on given testing input set x and testing labels y.

### train(x, y, validate_x, validate_y, batch_size=1, lr=0.001)

Train your model on training set x and training labels y. "validate_x" and "validate_y" are optional. If
given, after each batch training, absolute error on validation set is logged to the screen. In the future,
updates utilizing validation sets in training will come. Batch size and learning rate is self-explanatory.

### describe()

Prints a basic description of your model to the screen.

### compile(destination)

Compiles your model into mirror-MLgebra files and saves them in a directory named with your models name into
the specified directory with "destination".

## Methods

### readMnist(path1, path2, path3, path4, decimal)

Read MNISt database with decimal choice. "path1" and "path2" are training set and training labels.
"path3" and "path4" are test set and labels. Returns a 4-Vector-tuple. 

## Exceptions

### FileStructureError

This exception is raised when files extension is wrong.

### ConfigError

This function is raised when any problem with the model configuration occurs at any point of the structuring.


