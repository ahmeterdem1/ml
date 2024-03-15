from layer import *
from os.path import exists, isdir, dirname, abspath
from os import makedirs

PATH = dirname(abspath(__file__))

def integer(b):
    return int.from_bytes(b, byteorder="big")

def readMnist(path1, path2, path3, path4, decimal: bool = False):
    if not (isinstance(path1, str) or isinstance(path2, str) or isinstance(path3, str) or isinstance(path4, str)):
        raise ArgTypeError("Must be a string.")
    try:
        list1 = []
        with open(path1, "rb") as file:
            image_header = file.read(16)
            for k in range(60000):
                for count in range(1):
                    # ALl images are processed in this loop.
                    temp_image = []
                    for k in range(28):
                        temp = []
                        for l in range(28):
                            r = integer(file.read(1))
                            temp.append(r)
                        temp_image.append(Vector(*temp))
                    list1.append(Matrix(*temp_image))
            logger.info(f"Training set at {path1} read.")
    except Exception as e:
        logger.warning(f"File {path1} could not be read: {e}")
        return

    try:
        list2 = []
        with open(path2, "rb") as file:
            label_header = file.read(8)
            for k in range(60000):
                for l in range(1):
                    v = Vector.zero(10, decimal=decimal)
                    i = integer(file.read(1))
                    v[i] = 1
                    list2.append(v)
            logger.info(f"Training labels at {path2} read.")
    except Exception as e:
        logger.warning(f"File {path2} could not be read: {e}")
        return

    try:
        list3 = []
        with open(path3, "rb") as file:
            image_header = file.read(16)
            for k in range(10000):
                for count in range(1):
                    temp_image = []
                    for k in range(28):
                        temp = []
                        for l in range(28):
                            r = integer(file.read(1))
                            temp.append(r)
                        temp_image.append(Vector(*temp))
                    list3.append(Matrix(*temp_image))
            logger.info(f"Testing set at {path3} read.")
    except Exception as e:
        logger.warning(f"File {path3} could not be read: {e}")
        return

    try:
        list4 = []
        with open(path4, "rb") as file:
            label_header = file.read(8)
            for k in range(10000):
                for l in range(1):
                    v = Vector.zero(10, decimal=decimal)
                    i = integer(file.read(1))
                    v[i] = 1
                    list4.append(v)
            logger.info(f"Testing labels at {path4} read.")
    except Exception as e:
        logger.warning(f"File {path4} could not be read: {e}")
        return

    return list1, list2, list3, list4

class Model:

    def __init__(self, name: str,
                       loss: str = "crossentropy",
                       decimal: bool = False):
        self.name = name
        self.loss = loss.lower()
        self.decimal = decimal
        self.layers: List[Layer] = []

    def loadModel(self, path: str):
        if not exists(path):
            raise FileNotFoundError()

        with open(path, "r") as file:
            general_header = file.readline().replace("\n", "").split(":")
            layer_count = int(general_header[0])
            input_shape = int(general_header[1])

            if general_header[-1] == "1":
                self.decimal = True
                for k in range(layer_count):
                    header = file.readline().replace("\n", "").split(":")
                    if header[0] == "dense":
                        units = int(header[2])
                        layer = Dense(input_shape, units, header[1], template=True)
                        for l in range(units):
                            row = Vector(*[Decimal(m) for m in file.readline().replace("\n", "").split(":")])
                            layer.layer.append(Neuron(input_shape, template=True))
                            layer.layer[-1].weights = row
                        layer.bias = Vector(*[Decimal(m) for m in file.readline().replace("\n", "").split(":")])
                        layer.w_matrix = Matrix(*[n.weights for n in layer.layer])
                        self.layers.append(layer)
                    elif header[0] == "flatten":
                        self.layers.append(Flatten(int(header[-1])))
            else:
                self.decimal = False
                for k in range(layer_count):
                    header = file.readline().replace("\n", "").split(":")
                    if header[0] == "dense":
                        units = int(header[2])
                        layer = Dense(input_shape, units, header[1], template=True)
                        for l in range(units):
                            row = Vector(*[float(m) for m in file.readline().replace("\n", "").split(":")])
                            layer.layer.append(Neuron(input_shape, template=True))
                            layer.layer[-1].weights = row
                        layer.bias = Vector(*[float(m) for m in file.readline().replace("\n", "").split(":")])
                        layer.w_matrix = Matrix(*[n.weights for n in layer.layer])
                        self.layers.append(layer)
                    elif header[0] == "flatten":
                        self.layers.append(Flatten(int(header[-1])))

    def saveModel(self, path: str):
        if not isdir(path):
            raise NotADirectoryError()
        modelPath = path + "/" + self.name + ".model"
        if exists(modelPath):
            raise FileExistsError()

        with open(modelPath, "x") as file:
            file.write(f"{len(self.layers)}:{self.layers[0].units}:{'1' if self.decimal else '0'}\n")
            for layer in self.layers:
                file.write(f"{layer.name}:{layer.activation}:{layer.units}\n")
                for neuron in layer.layer:
                    file.write(":".join([str(k) for k in neuron.weights.values]) + "\n")
                if layer.bias:
                    file.write(":".join([str(k) for k in layer.bias.values]) + "\n")

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def structure(self, s: List[Layer]):
        self.layers = [layer for layer in s]

    def produce(self, input: Union[Tensor, Matrix, Vector]):
        output = input
        check = False
        for k in range(len(self.layers)):
            if check:
                check = False
                continue
            elif self.layers[k].name == "flatten":
                output = self.layers[k].startForward(input)
                output = self.layers[k + 1].startForward(output)
                check = True
            elif self.layers[k].name == "dense":
                output = self.layers[k].forward(output)
        return output

    def evaluate(self, x: Union[List[Vector], List[Matrix], Matrix, Tensor],
                       y: Union[List[Vector], List[Matrix], Matrix, Tensor]):
        if self.loss == "crossentropy":
            error = Vector.zero(self.layers[-1].units, self.decimal)

            for k in range(len(x)):
                error += y[k].dot(Vector(*[ln(k) for k in self.produce(x[k]).values]))
            error /= len(x)

            return -error.cumsum()


    def train(self, x: Union[List[Vector], List[Matrix], Matrix, Tensor],
                    y: Union[List[Vector], List[Matrix], Matrix, Tensor],
                    validation_x: Union[List[Vector], List[Matrix], Matrix, Tensor, None] = None,
                    validation_y: Union[List[Vector], List[Matrix], Matrix, Tensor, None] = None,
                    batch_size: int = 1,
                    lr: Union[int, float, Decimal] = 0.01
              ):

        if self.loss == "crossentropy":
            count = len(x) // batch_size
            for k in range(count - 1):
                error = Vector.zero(self.layers[-1].units, self.decimal)

                logger.info(f"{(k / count):.2f}% done.")
                for l in range(batch_size):
                    error += y[k * batch_size + l] - self.produce(x[k * batch_size + l])

                error /= batch_size
                error = self.layers[-1].startBackward(error, self.layers[-2].output, lr)

                for i in range(len(self.layers) - 2, 0, -1):
                    error_sum = []
                    if self.layers[i - 1].name == "flatten":
                        continue
                    elif self.layers[i].name == "dense" and self.layers[i + 1].name == "dense":
                        for j in range(self.layers[i].units):
                            e_sum = 0
                            for m in range(self.layers[i + 1].units):
                                e_sum += self.layers[i + 1].layer[m].weights[j] * error[m]
                            error_sum.append(e_sum)

                        error = Vector(*error_sum)
                        self.layers[i].backward(error, self.layers[i - 1].output, lr)

                for i in range(1, len(self.layers)):
                    if self.layers[i - 1].name != "flatten":
                        self.layers[i].update()

                if validation_x:
                    logger.info(f"Error at batch {k} with {self.loss}: {self.evaluate(validation_x, validation_y)}")

            # Last Batch
            error = Vector.zero(self.layers[-1].units, self.decimal)
            for k in range(len(x) - (count - 1) * batch_size):
                error += y[(count - 1) * batch_size + k] - self.produce(x[(count - 1) * batch_size + k])

            error /= len(x) - (count - 1) * batch_size
            error = self.layers[-1].startBackward(error, self.layers[-2].output, lr)
            for i in range(len(self.layers) - 2, 0, -1):
                error_sum = []
                if self.layers[i - 1].name == "flatten":
                    continue
                elif self.layers[i].name == "dense" and self.layers[i + 1].name == "dense":
                    for j in range(self.layers[i].units):
                        e_sum = 0
                        for m in range(self.layers[i + 1].units):
                            e_sum += self.layers[i + 1].layer[m].weights[j] * error[m]
                        error_sum.append(e_sum)

                    error = Vector(*error_sum)
                    self.layers[i].backward(error, self.layers[i - 1].output, lr)

            for i in range(1, len(self.layers)):
                if self.layers[i - 1].name != "flatten":
                    self.layers[i].update()

            if validation_x:
                logger.info(f"Error at last batch with {self.loss}: {self.evaluate(validation_x, validation_y)}")

    def describe(self):
        print(f"Model: {self.name}")
        for layer in self.layers:
            print(f"-> {layer.name} | {layer.units} | {layer.activation}")

    def compile(self, destination: str):
        if not isdir(destination):
            raise NotADirectoryError()

        try:
            with open(f"{PATH}/spy/neuron.spy", "r") as file:
                neuron_file = file.read()
        except Exception as e:
            logger.warning(f"neuron.spy file could not be read at directory {PATH}/spy/: {e}")
            return

        neuron_file = neuron_file.split("[NEURON]")
        neuron_header = neuron_file[0]
        neuron_template = neuron_file[1].replace("[DECIMAL]", str(self.decimal))
        del neuron_file

        try:
            with open(f"{PATH}/spy/layer.spy", "r") as file:
                layer_file = file.read()
        except Exception as e:
            logger.warning(f"layer.spy file could not be read at directory {PATH}/spy/: {e}")
            return

        layer_file = layer_file.split("[DENSE]")
        layer_header = layer_file[0]
        layer_template = layer_file[1].replace("[DECIMAL]", str(self.decimal))
        del layer_file

        try:
            with open(f"{PATH}/spy/model.spy", "r") as file:
                model_file = file.read()
        except Exception as e:
            logger.warning(f"model.spy file could not be read at directory {PATH}/spy/: {e}")
            return

        model_file = model_file.split("[MODEL]")
        model_header = model_file[0]
        model_template = model_file[1].replace("[DECIMAL]", str(self.decimal)) \
            .replace("[MODEL_NAME]", f"'{self.name}'").replace("[LOSS_FUNCTION]", f"'{self.loss}'")

        del model_file

        passed_flatten = False
        index_count = 0
        for index, layer in enumerate(self.layers):

            if layer.name == "flatten":
                passed_flatten = True
                index_count += 1
                model_template = model_template.replace("[START_PRODUCE]", f"output = output.reshape({layer.units})"
                                                           f"\n        [START_PRODUCE]")
                continue

            neuron_temporary = neuron_template.replace("[LAYER_ORDER]", str(index))
            layer_temporary = layer_template.replace("[LAYER_ORDER]", str(index)) \
                    .replace("[LAYER_TYPE_NAME]", f"'{layer.name}'").replace("[LEAK]", str(layer.leak)) \
                    .replace("[CUTOFF]", str(layer.cutoff)).replace("[ACTIVATION_NAME]", f"'{layer.activation}'") \
                    .replace("[UNITS]", str(layer.units))

            if index == 0:
                layer_temporary = layer_temporary.replace("[INPUT_SHAPE]", str(layer.units))
            else:
                layer_temporary = layer_temporary.replace("[INPUT_SHAPE]", str(self.layers[index - 1].units))

            # Neuron generation
            if layer.generation_info[0] == "zero":
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]", f".zero({layer.generation_info[-1]}, {self.decimal})")

            elif layer.generation_info[0] == "one":
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]", f".one({layer.generation_info[-1]}, {self.decimal})")

            elif layer.generation_info[0] == "uniform" or layer.generation_info[0] == "flat":
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]",
                                         f".randVfloat({layer.generation_info[-1]}, "
                                         f"{layer.generation_info[1]}, "
                                         f"{layer.generation_info[2]}, {self.decimal})")

            elif layer.generation_info[0] == "naive":
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]",
                                         f".randVgauss({layer.generation_info[-1]}, "
                                         f"{layer.generation_info[1]}, "
                                         f"{layer.generation_info[2]}, {self.decimal})")

            elif layer.generation_info[0] == "uxavier":
                limiter = sqrt(6 / (layer.generation_info[1] + layer.generation_info[-1]))
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]",
                                         f".randVgauss({layer.generation_info[-1]}, "
                                         f"{-limiter}, "
                                         f"{limiter}, {self.decimal})")

            elif layer.generation_info[0] == "nxavier" or layer.generation_info[0] == "xavier":
                sigma = sqrt(2 / (layer.generation_info[1] + layer.generation_info[-1]))
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]",
                                         f".randVgauss({layer.generation_info[-1]}, "
                                         f"0, {sigma}, {self.decimal})")

            elif layer.generation_info[0] == "he":
                sigma = 2 / sqrt(layer.generation_info[1])
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]",
                                         f".randVgauss({layer.generation_info[-1]}, "
                                         f"0, {sigma}, {self.decimal})")

            else:
                neuron_temporary = neuron_temporary.replace("[WEIGHT_GENERATE]", f"(*[{layer.generation_info[1]}"
                                                              f" for k in range({layer.generation_info[-1]})])")

            # Defining the derivatives and the activation function
            if layer.activation == "minmax":
                layer_temporary = layer_temporary.replace("[ACTIVATION]", ".minmax()") \
                    .replace("[DERIVATIVE]", "self.output[i] * (1 - self.output[i])")
                # So that the derivative is defined. No other purpose

            elif layer.activation == "relu":
                layer_temporary = layer_temporary.replace("[ACTIVATION]", ".relu(self.leak, self.cutoff)") \
                    .replace("[DERIVATIVE]", "deriv_relu(self.output[i], self.leak, self.cutoff)")

            elif layer.activation == "sigmoid":
                layer_temporary = layer_temporary.replace("[ACTIVATION]", ".sigmoid(1, self.cutoff)") \
                    .replace("[DERIVATIVE]", "self.output[i] * (1 - self.output[i])")

            elif layer.activation == "softmax":
                layer_temporary = layer_temporary.replace("[ACTIVATION]", ".softmax()") \
                    .replace("[DERIVATIVE]", "self.output[i] * (1 - self.output[i])")

            else:
                layer_temporary = layer_temporary.replace("[ACTIVATION]", "") \
                    .replace("[DERIVATIVE]", "1")

            # Layer bias generation
            if layer.generation_info[-2] == "zero":
                layer_temporary = layer_temporary.replace("[BIAS_GENERATE]", ".zero(self.units, self.decimal)")
            elif layer.generation_info[-2] == "one":
                layer_temporary = layer_temporary.replace("[BIAS_GENERATE]", ".one(self.units, self.decimal)")
            elif layer.generation_info[-2] == "naive":
                layer_temporary = layer_temporary.replace("[BIAS_GENERATE]", f".randVgauss(self.units, "
                                                           f"{layer.generation_info[3]}, "
                                                           f"{layer.generation_info[4]}, self.decimal)")
            elif layer.generation_info[-2] == "flat":
                layer_temporary = layer_temporary.replace("[BIAS_GENERATE]", f".randVfloat(self.units, "
                                                           f"{layer.generation_info[3]}, "
                                                           f"{layer.generation_info[4]}, self.decimal)")
            else:
                layer_temporary.replace("[BIAS_GENERATE]", f"([{layer.generation_info[3]} for k in range(self.units)])")

            if passed_flatten:
                model_template = model_template.replace("[START_PRODUCE]", f"output = self.layers[{index - index_count}].startForward(output)"
                                                          f"\n        [START_PRODUCE]")
                passed_flatten = False
            else:
                model_template = model_template.replace("[START_PRODUCE]", f"output = self.layers[{index - index_count}].forward(output)"
                                                          f"\n        [START_PRODUCE]")

            if len(self.layers) - 1 - index_count > index > 0:
                model_template = model_template.replace("[BACKPROPAGATE]", f"""[BACKPROPAGATE]\n
            error_sum = []
            for j in range(self.layers[{index + 1 - index_count}].units):
                e_sum = 0
                for m in range(self.layers[{index + 2 - index_count}].units):
                    e_sum += self.layers[{index + 2 - index_count}].layer[m].weights[j] * error[m]
                error_sum.append(e_sum)

            error = Vector(*error_sum)
            self.layers[{index + 1 - index_count}].backward(error, self.layers[{index - index_count}].output, lr)\n""")
                model_template = model_template.replace("[LASTPROPAGATE]", f"""[LASTPROPAGATE]\n
        error_sum = []
        for j in range(self.layers[{index + 1 - index_count}].units):
            e_sum = 0
            for m in range(self.layers[{index + 2 - index_count}].units):
                e_sum += self.layers[{index + 2 - index_count}].layer[m].weights[j] * error[m]
            error_sum.append(e_sum)

        error = Vector(*error_sum)
        self.layers[{index + 1 - index_count}].backward(error, self.layers[{index - index_count}].output, lr)\n""")

            model_template = model_template.replace("[READ]", rf"""
            header = file.readline().replace("\n", "").split(":")
            units = int(header[2])
            layer = Dense{index}(input_shape, units, header[1], template=True)
            for l in range(units):
                row = Vector(*[[NUMBER_TYPE](m) for m in file.readline().replace("\n", "").split(":")])
                layer.layer.append(Neuron{index}(input_shape, template=True))
                layer.layer[-1].weights = row
            layer.bias = Vector(*[[NUMBER_TYPE](m) for m in file.readline().replace("\n", "").split(":")])
            layer.w_matrix = Matrix(*[n.weights for n in layer.layer])
            self.layers.append(layer)
            
            [READ]""")

            neuron_header += neuron_temporary
            layer_header += layer_temporary

        model_template = model_template.replace("[START_PRODUCE]", "").replace("[BACKPROPAGATE]", "") \
            .replace("[LASTPROPAGATE]", "").replace("[READ]", "")

        if self.decimal:
            model_template = model_template.replace("[NUMBER_TYPE]", "Decimal")
        else:
            model_template = model_template.replace("[NUMBER_TYPE]", "float")

        makedirs(f"{destination}/{self.name}")

        with open(f"{PATH}/exception.py", "r") as file:
            exc = file.read()

        with open(f"{destination}/{self.name}/exception.py", "w") as file:
            file.write(exc)

        with open(f"{destination}/{self.name}/neuron.py", "w") as file:
            file.write(neuron_header)

        with open(f"{destination}/{self.name}/layer.py", "w") as file:
            file.write(layer_header)

        with open(f"{destination}/{self.name}/model.py", "w") as file:
            file.write(model_header + model_template)

        logger.info(f"Model compiled and saved at {destination}/{self.name}")


