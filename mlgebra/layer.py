from neuron import *

class Layer:
    name: str
    layer: List[Neuron]
    units: int
    bias: Vector
    next_bias: Vector
    output: Vector
    w_matrix: Matrix
    activation: str
    leak: Union[int, float, Decimal]
    cutoff: Union[int, float, Decimal]
    decimal: bool
    generation_info: List

    # This class is created so that when i want to create
    # other types of layers like Convolution, RNN, etc. it
    # will be easier and more modular.

    def __init__(self):
        raise NotImplementedError()

    def forward(self, input_vector):
        # This will return the "a" vector produced by the layer.
        raise NotImplementedError()

    def startForward(self, input_vector):
        raise NotImplementedError()

    def backward(self, deltas, inputs, lr):
        # This will return the vector of delta's calculated for the weights.
        raise NotImplementedError()

    def startBackward(self, deltas, inputs, lr):
        # This is only for the last layer in the model.
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

class Dense(Layer):

    def __init__(
                 self,
                 input_shape: int,
                 units: int,
                 activation: str = "relu",
                 bias: str = "zero",
                 initialization: str = "uxavier",
                 decimal: bool = False,
                 low: Union[int, float, Decimal] = -1,
                 high: Union[int, float, Decimal] = 1,
                 bias_low: Union[int, float, Decimal] = -1,
                 bias_high: Union[int, float, Decimal] = 1,
                 cutoff: Union[int, float, Decimal] = 0,
                 leak: Union[int, float, Decimal] = 0,
                 template: bool = False
                 ):

        self.name = "dense"  # So that we won't need to do isinstance() calls.
        self.units = units
        self.decimal = decimal

        self.activation = activation.lower()
        bias = bias.lower()
        initialization = initialization.lower()

        self.leak = leak
        self.cutoff = cutoff


        if template:
            self.next_bias = Vector.zero(self.units, self.decimal)
            self.output = Vector.zero(units, self.decimal)
            self.layer = []
        else:
            if "xavier" in initialization or initialization == "he":
                low = self.units

            self.layer = [Neuron(input_shape, initialization, self.decimal, low, high) for k in range(self.units)]  # A list of nodes
            self.w_matrix = Matrix(*[n.weights for n in self.layer])
            self.generation_info = (initialization, low, high, bias_low, bias_high, bias, input_shape)

            if bias == "zero":
                self.bias = Vector.zero(self.units, self.decimal)
            elif bias == "one":
                self.bias = Vector.one(self.units, self.decimal)
            elif bias == "naive":
                self.bias = Vector.randVgauss(self.units, bias_low, bias_high, self.decimal)
            elif bias == "flat":
                self.bias = Vector.randVfloat(self.units, bias_low, bias_high, self.decimal)
            else:  # includes bias == "constant"
                self.bias = Vector([bias_low for k in range(self.units)])

            self.next_bias = Vector.zero(self.units, self.decimal)
            self.output = Vector.zero(units, self.decimal)

    def forward(self, input_vector: Vector):
        # This will be called from the "Model" class.

        z = self.w_matrix * input_vector + self.bias

        if self.activation == "minmax":
            return z.minmax()
        elif self.activation == "relu":
            return z.relu(self.leak, self.cutoff)
        elif self.activation == "sigmoid":
            return z.sig(1, self.cutoff)
        elif self.activation == "softmax":
            return z.softmax()
        else:
            return z

    def startForward(self, input_vector):
        if self.activation == "minmax":
            return input_vector.minmax()
        elif self.activation == "relu":
            return input_vector.relu(self.leak, self.cutoff)
        elif self.activation == "sigmoid":
            return input_vector.sig(1, self.cutoff)
        elif self.activation == "softmax":
            return input_vector.softmax()
        else:
            return input_vector

    def backward(self, deltas: Vector, inputs: Vector, lr: Union[int, float, Decimal]):
        # deltas will have error sums as components.
        # Proper calculations will be done in the "Model" class.
        output = Vector()

        if self.activation == "relu":
            for i in range(self.units):
                delta = deltas[i] * deriv_relu(self.output[i], self.leak, self.cutoff)
                output.append(delta)
                self.layer[i].next_weights = self.layer[i].weights - lr * delta * inputs[i]
        elif self.activation == "sigmoid":
            for i in range(self.units):
                delta = deltas[i] * self.output[i] * (1 - self.output[i])
                output.append(delta)
                self.layer[i].next_weights = self.layer[i].weights - lr * delta * inputs[i]
        self.next_bias = self.bias - deltas * lr
        return output

    def startBackward(self, deltas: Vector, inputs: Vector, lr: Union[int, float, Decimal]):
        for i in range(self.units):
            self.layer[i].next_weights = self.layer[i].weights - lr * deltas[i] * inputs[i]
        self.next_bias = self.bias - deltas * lr
        return deltas

    def update(self):
        vlist = []
        for neuron in self.layer:
            neuron.weights = neuron.next_weights
            vlist.append(neuron.next_weights)
            neuron.next_weights = Vector.zero(self.units, self.decimal)
        self.w_matrix = Matrix(*vlist)
        self.bias = self.next_bias

class Flatten(Layer):

    def __init__(self, shape: int):
        self.name = "flatten"
        self.units = shape
        self.activation = "pass-through"
        self.layer = []
        self.bias = []
        self.w_matrix = Matrix()

    def startForward(self, input_vector: Union[Vector, Matrix, Tensor]):
        if isinstance(input_vector, Tensor):
            return input_vector.flatten()
        elif isinstance(input_vector, Matrix):
            return input_vector.reshape(self.units)
        else:
            return input_vector

    def forward(self, input_vector):
        if isinstance(input_vector, Tensor):
            return input_vector.flatten()
        elif isinstance(input_vector, Matrix):
            return input_vector.reshape(self.units)
        else:
            return input_vector


