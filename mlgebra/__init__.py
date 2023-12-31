from vectorgebra import *
import os
import multiprocessing as mp

class FileStructureError(Exception):

    def __init__(self, hint: str = ""):
        super().__init__(f"File structured incorrectly{': ' + hint if hint else ''}")

class ConfigError(Exception):

    def __init__(self, hint: str = ""):
        super().__init__(f"Model configuration invalid{': ' + hint if hint else ''}")

class Node:

    def __init__(self):
        self.w = None

class Model:

    def __init__(self, name: str = "", decimal: bool = False, multiprocess: bool = False):
        if os.path.exists(f"./{name}.weights"):
            raise FileNotFoundError()
        self.name = name
        self.layers = []
        self.errors = []
        self.bias = []
        self.last_output = []
        self.w_matrices = []
        self.decimal = decimal
        self.output_func = "softmax"
        self.activation = "relu"
        self.input_func = "minmax"
        self.details = {"cutoff": 0, "leak": 0}
        self.multiprocess = multiprocess
        self.templist = []

    def __str__(self):
        return (f"Model Name: {self.name}\nModel Structre: {'x'.join([str(len(k)) for k in self.layers])}\n"
                f"Input Processing: {self.input_func}\nActivation Function: {self.activation}\n"
                f"Output Processing: {self.output_func}\n")

    def addDense(self, amount: int = 1):
        self.layers.append([Node() for k in range(amount)])

    def saveModel(self):
        with open(f"{self.name}.weights", "x") as file:
            layer_count = 0
            file.write(f"-1:b:" + ",".join([str(k) for k in self.bias[0]]) + "\n")
            for layer in self.layers[1:]:
                node_count = 0
                for node in layer:
                    file.write(f"{layer_count}:{node_count}:" + ",".join([str(k) for k in node.w.values]) + "\n")
                    node_count += 1
                file.write(f"{layer_count}:b:" + ",".join([str(k) for k in self.bias[layer_count + 1]]) + "\n")
                layer_count += 1

    def readWeightFile(self, path: str = ""):
        if not path.endswith(".weights"): raise FileStructureError("Incorrect extension.")
        with open(path, "r") as file:
            all = file.read()

        lines = all.split("\n")
        for line in lines:
            if line == "":
                continue
            parts = line.split(":")
            i = int(parts[0])

            if parts[1] == "b":
                # "i" is not important here.
                weights = parts[2].split(",")
                self.bias.append(Vector(*[float(k) for k in weights]))
            else:
                j = int(parts[1])
                weights = parts[2].split(",")

                self.layers[i + 1][j].w = Vector(*[float(k) for k in weights])

        for layer in self.layers[1:]:
            w_list = []
            for node in layer:
                w_list.append(node.w)
            self.w_matrices.append(Matrix(*w_list))

        logger.info(f"Weight file read at {path}")

    def readMNIST(self, path1, path2, path3, path4):
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
                        v = Vector.zero(10, decimal=self.decimal)
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
                        v = Vector.zero(10, decimal=self.decimal)
                        i = integer(file.read(1))
                        v[i] = 1
                        list4.append(v)
                logger.info(f"Testing labels at {path4} read.")
        except Exception as e:
            logger.warning(f"File {path4} could not be read: {e}")
            return

        return list1, list2, list3, list4

    def finalize(self, generation: str = "flat", a=-2, b=2):
        generation = generation.lower()

        if generation == "uxavier":
            for i in range(1, len(self.layers)):
                w_list = []
                limiter = sqrt(6 / (len(self.layers[i - 1]) + len(self.layers[i])))
                for node in self.layers[i]:
                    node.w = Vector.randVfloat(dim=len(self.layers[i - 1]), a=-limiter, b=limiter, decimal=self.decimal)
                    w_list.append(node.w)
                self.w_matrices.append(Matrix(*w_list))
            logger.debug("Weights are initialized via Uniform Xavier/Glorot initialization.")
            return
        if generation == "nxavier":
            for i in range(1, len(self.layers)):
                w_list = []
                sigma = sqrt(2 / (len(self.layers[i - 1]) + len(self.layers[i])))
                for node in self.layers[i]:
                    node.w = Vector.randVgauss(dim=len(self.layers[i - 1]), mu=0, sigma=sigma, decimal=self.decimal)
                    w_list.append(node.w)
                self.w_matrices.append(Matrix(*w_list))
            logger.debug("Weights are initialized via Normal Xavier/Glorot initialization.")
            return
        if generation == "he":
            for i in range(1, len(self.layers)):
                w_list = []
                for node in self.layers[i]:
                    node.w = Vector.randVgauss(dim=len(self.layers[i - 1]), mu=0, sigma=(2/sqrt(len(self.layers[i - 1]))), decimal=self.decimal)
                    w_list.append(node.w)
                self.w_matrices.append(Matrix(*w_list))
            logger.debug("Weights are initialized via He initialization.")
            return
        if generation == "naive":
            for i in range(1, len(self.layers)):
                w_list = []
                for node in self.layers[i]:
                    # First other argument is the standard deviation
                    node.w = Vector.randVgauss(dim=len(self.layers[i - 1]), mu=0, sigma=a, decimal=self.decimal)
                    w_list.append(node.w)
                self.w_matrices.append(Matrix(*w_list))
            logger.debug(f"Weights are initialized via naive (default normal) initialization; sigma: {a}")
            return
        if generation == "flat":
            for i in range(1, len(self.layers)):
                w_list = []
                for node in self.layers[i]:
                    node.w = Vector.randVfloat(dim=len(self.layers[i - 1]), a=a, b=b, decimal=self.decimal)
                    w_list.append(node.w)
                self.w_matrices.append(Matrix(*w_list))
            logger.debug(f"Weights are initialized via flat (uniform float) initialization; range: [{a}, {b})")
            return
        raise ConfigError("Incorrect choice.")

    def includeBias(self, generation: str = "flat", a=-2, b=2):
        generation = generation.lower()

        if generation == "flat":
            for i in range(len(self.layers)):
                self.bias.append(Vector.randVfloat(dim=len(self.layers[i]), a=a, b=b, decimal=self.decimal) / (i + 1))
            logger.debug(f"Biases are initialized via flat (uniform float) initialization; range: [{a}, {b})")
            return
        if generation == "zero":
            for i in range(len(self.layers)):
                self.bias.append(Vector.zero(dim=len(self.layers[i]), decimal=self.decimal))
            logger.debug("Biases are initialized to all zero.")
            return
        if generation == "constant":
            for i in range(len(self.layers)):
                self.bias.append(Vector.one(dim=len(self.layers[i]), decimal=self.decimal) * a)
            logger.debug(f"Biases are initialized to all constant value: {a}")
            return
        raise ConfigError("Incorrect generation choice.")

    def configureMethods(self, **kwargs):
        choices = ["minmax", "softmax", "sigmoid", "relu"]
        for key in ["input", "output", "activator", "cutoff", "leak"]:

            try:
                if key == "cutoff":
                    if not (isinstance(kwargs[key], int) or isinstance(kwargs[key], float) or isinstance(kwargs[key], Decimal)
                        or isinstance(kwargs[key], Infinity)): raise ArgTypeError("Must be a numerical value.")
                    self.details["cutoff"] = kwargs[key]
                    continue

                if key == "leak":
                    if not (isinstance(kwargs[key], int) or isinstance(kwargs[key], float) or isinstance(kwargs[key], Decimal)
                        or isinstance(kwargs[key], Infinity)): raise ArgTypeError("Must be a numerical value.")
                    self.details["leak"] = kwargs[key]
                    continue

                if kwargs[key] not in choices: raise ConfigError("Wrong choice.")
                if key == "input":
                    self.input_func = kwargs[key]
                elif key == "output":
                    self.output_func = kwargs[key]
                elif key == "activator":
                    self.activation = kwargs[key]
            except KeyError as e:
                logger.info(f"Key not found, ignoring the key: {e}")

    def updateMatrices(self):
        self.w_matrices = []
        for layer in self.layers[1:]:
            temp = []
            for node in layer:
                temp.append(node.w)
            self.w_matrices.append(Matrix(*temp))

    def _produce(self, d, b: bool = True):
        if isinstance(d, Matrix):
            dims = len(d.values) * len(d.values[0])
            temp = d.reshape(dims)
        else:
            temp = d

        if b:
            # INPUT
            if self.input_func == "minmax":
                temp = temp.minmax()
            elif self.input_func == "sigmoid":
                temp = temp.sig(cutoff=self.details["cutoff"])
            elif self.input_func == "relu":
                temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

            temp += self.bias[0]
            counter = 1

            # PROCESSING
            if self.activation == "relu":
                for matrix in self.w_matrices[:-1]:
                    if self.w_matrices.index(matrix) == 0:
                        temp = matrix * temp
                        temp += self.bias[counter]
                    else:
                        temp = matrix * temp
                        temp += self.bias[counter]
                        temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])
                    counter += 1

            elif self.activation == "sigmoid":
                for matrix in self.w_matrices[:-1]:
                    if self.w_matrices.index(matrix) == 0:
                        temp = matrix * temp
                        temp += self.bias[counter]
                    else:
                        temp = matrix * temp
                        temp += self.bias[counter]
                        temp = temp.sig(cutoff=self.details["cutoff"])
                    counter += 1

            # OUTPUT
            if self.output_func == "softmax":
                temp = self.w_matrices[-1] * temp
                temp += self.bias[counter]
                temp = temp.softmax()
            elif self.output_func == "minmax":
                temp = self.w_matrices[-1] * temp
                temp += self.bias[counter]
                temp = temp.minmax()
            elif self.output_func == "sigmoid":
                temp = self.w_matrices[-1] * temp
                temp += self.bias[counter]
                temp = temp.sig(cutoff=self.details["cutoff"])
            elif self.output_func == "relu":
                temp = self.w_matrices[-1] * temp
                temp += self.bias[counter]
                temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

        else:
            # INPUT
            if self.input_func == "minmax":
                temp = temp.minmax()
            elif self.input_func == "sigmoid":
                temp = temp.sig(cutoff=self.details["cutoff"])
            elif self.input_func == "relu":
                temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

            # PROCESSING
            if self.activation == "relu":
                for matrix in self.w_matrices[:-1]:
                    if self.w_matrices.index(matrix) == 0:
                        temp = matrix * temp
                    else:
                        temp = matrix * temp
                        temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

            elif self.activation == "sigmoid":
                for matrix in self.w_matrices[:-1]:
                    if self.w_matrices.index(matrix) == 0:
                        temp = matrix * temp
                    else:
                        temp = matrix * temp
                        temp = temp.sig(cutoff=self.details["cutoff"])

            # OUTPUT
            if self.output_func == "softmax":
                temp = self.w_matrices[-1] * temp
                temp = temp.softmax()
            elif self.output_func == "minmax":
                temp = self.w_matrices[-1] * temp
                temp = temp.minmax()
            elif self.output_func == "sigmoid":
                temp = self.w_matrices[-1] * temp
                temp = temp.sig(cutoff=self.details["cutoff"])
            elif self.output_func == "relu":
                temp = self.w_matrices[-1] * temp
                temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

        return temp

    def singleTrain(self, d, label, learning_rate):
        self.last_output = [Vector.zero(len(k), decimal=self.decimal) for k in self.layers]
        error_list = []
        error = Vector.zero(len(self.layers[-1]), decimal=self.decimal)

        if self.bias:
            temp = self._produce(d, True)
            error += temp - label
            error_list.append(error)

            previous_delta = False
            new_weights = []
            new_biases = []
            if self.activation == "relu":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(
                        deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum)
                error_list.append(Vector(*last_delta))
            elif self.activation == "sigmoid":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = o * (1 - o) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(o * (1 - o) * error_sum)
                error_list.append(Vector(*last_delta))

            for i in range(len(self.layers)):
                new_biases.append(self.bias[i] + learning_rate * error_list[-1 - i])
            for i in range(1, len(self.layers)):
                for j in range(len(self.layers[i])):
                    self.layers[i][j].w = new_weights[i - 1][j]
            self.bias = [v for v in new_biases]

        else:
            temp = self._produce(d, False)
            error += temp - label
            error_list.append(error)

            previous_delta = False
            new_weights = []
            if self.activation == "relu":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(
                        deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum)
                error_list.append(Vector(*last_delta))
            elif self.activation == "sigmoid":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = o * (1 - o) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(o * (1 - o) * error_sum)
                error_list.append(Vector(*last_delta))

        self.updateMatrices()

    def train(self, d, label, learning_rate):
        global _temporary
        self.last_output = [Vector.zero(len(k), decimal=self.decimal) for k in self.layers]
        error_list = []
        error = Vector.zero(len(self.layers[-1]), decimal=self.decimal)
        length = len(d)

        if self.bias:
            if self.multiprocess:
                p_count = int(log2(len(d)) // 1)
                pool = []
                item_count = int((length // p_count) + 1)
                manager = mp.Manager()
                temporary = manager.dict()
                for p in range(p_count - 1):
                    pool.append(mp.Process(target=_multiproduce,
                                           args=[self, d[p * item_count:(p + 1) * item_count], True, temporary, p]))
                    pool[-1].start()
                pool.append(
                    mp.Process(target=_multiproduce, args=[self, d[(p_count - 1) * item_count:], True, temporary, p_count]))
                pool[-1].start()
                logger.info(f"{p_count} processes started.")
                for p in pool:
                    p.join()
                logger.info("All processes returned")
                for k, temp in temporary.items():
                    error += temp - label[k]
                error /= length
                error_list.append(error)
            else:
                for i, data in enumerate(d):
                    temp = self._produce(data, False)
                    error += temp - label[i]
                    error /= len(d)
                error_list.append(error)
            previous_delta = False
            new_weights = []
            new_biases = []
            if self.activation == "relu":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = deriv_relu(o, cutoff=self.details["cutoff"],
                                               leak=self.details["leak"]) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(
                        deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum)
                error_list.append(Vector(*last_delta))


            elif self.activation == "sigmoid":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = o * (1 - o) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(o * (1 - o) * error_sum)
                error_list.append(Vector(*last_delta))

            for i in range(len(self.layers)):
                new_biases.append(self.bias[i] + learning_rate * error_list[-1 - i])
            for i in range(1, len(self.layers)):
                for j in range(len(self.layers[i])):
                    self.layers[i][j].w = new_weights[i - 1][j]
            self.bias = [v for v in new_biases]

        else:
            if self.multiprocess:
                p_count = int(log2(len(d)) // 1)
                pool = []
                item_count = int((length // p_count) + 1)
                manager = mp.Manager()
                temporary = manager.dict()
                for p in range(p_count - 1):
                    pool.append(mp.Process(target=_multiproduce,
                                           args=[self, d[p * item_count:(p + 1) * item_count], True, temporary, p]))
                    pool[-1].start()
                pool.append(
                    mp.Process(target=_multiproduce,
                               args=[self, d[(p_count - 1) * item_count:], True, temporary, p_count]))
                pool[-1].start()

                for p in pool:
                    p.join()
                for k, temp in temporary.items():
                    error += temp - label[k]
                error /= length
                error_list.append(error)
            else:
                for i, data in enumerate(d):
                    temp = self._produce(data, False)
                    error += temp - label[i]
                    error /= len(d)
                error_list.append(error)

            previous_delta = False
            new_weights = []
            if self.activation == "relu":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(
                        deriv_relu(o, cutoff=self.details["cutoff"], leak=self.details["leak"]) * error_sum)
                error_list.append(Vector(*last_delta))

            elif self.activation == "sigmoid":
                for index in range(len(self.layers) - 1, 0, -1):
                    list_for_prev_delta = []
                    new_weights.insert(0, [])
                    for node_index in range(len(self.layers[index])):
                        node = self.layers[index][node_index]
                        o = self.last_output[index][node_index]

                        if not previous_delta:
                            # OUTPUT LAYER
                            delta = error_list[-1][node_index]
                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                        else:
                            # HIDDENS EXCEPT INPUT
                            error_sum = 0
                            counter = 0
                            for prev_node in self.layers[index + 1]:
                                error_sum += prev_node.w[node_index] * error_list[-1][counter]
                                counter += 1
                            delta = o * (1 - o) * error_sum

                            list_for_prev_delta.append(delta)
                            new_weights[0].append(node.w + learning_rate * delta * self.last_output[index - 1])
                    error_list.append(Vector(*list_for_prev_delta))
                    previous_delta = True

                last_delta = []
                # INPUT LAYER
                for i in range(len(self.layers[0])):
                    o = self.last_output[0][i]
                    error_sum = 0
                    counter = 0
                    for prev_node in self.layers[1]:
                        error_sum += prev_node.w[i] * error_list[-1][counter]
                        counter += 1
                    last_delta.append(o * (1 - o) * error_sum)
                error_list.append(Vector(*last_delta))

        self.updateMatrices()

    def produce(self, d):
        if isinstance(d, Matrix):
            dims = len(d.values) * len(d.values[0])
            temp = d.reshape(dims)
        else:
            temp = d

        # INPUT
        if self.input_func == "minmax":
            temp = temp.minmax()
        elif self.input_func == "sigmoid":
            temp = temp.sig(cutoff=self.details["cutoff"])
        elif self.input_func == "relu":
            temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

        if self.bias:
            temp += self.bias[0]
        counter = 1

        # PROCESSING
        if self.activation == "relu":
            for matrix in self.w_matrices[:-1]:
                if self.w_matrices.index(matrix) == 0:
                    temp = matrix * temp
                    if self.bias:
                        temp += self.bias[counter]
                else:
                    temp = matrix * temp
                    if self.bias:
                        temp += self.bias[counter]
                    temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])
                counter += 1

        elif self.activation == "sigmoid":
            for matrix in self.w_matrices[:-1]:
                if self.w_matrices.index(matrix) == 0:
                    temp = matrix * temp
                    if self.bias:
                        temp += self.bias[counter]
                else:
                    temp = matrix * temp
                    if self.bias:
                        temp += self.bias[counter]
                    temp = temp.sig(cutoff=self.details["cutoff"])
                counter += 1

        # OUTPUT
        if self.output_func == "softmax":
            temp = self.w_matrices[-1] * temp
            if self.bias:
                temp += self.bias[counter]
            temp = temp.softmax()
        elif self.output_func == "minmax":
            temp = self.w_matrices[-1] * temp
            if self.bias:
                temp += self.bias[counter]
            temp = temp.minmax()
        elif self.output_func == "sigmoid":
            temp = self.w_matrices[-1] * temp
            if self.bias:
                temp += self.bias[counter]
            temp = temp.sig(cutoff=self.details["cutoff"])
        elif self.output_func == "relu":
            temp = self.w_matrices[-1] * temp
            if self.bias:
                temp += self.bias[counter]
            temp = temp.relu(cutoff=self.details["cutoff"], leak=self.details["leak"])

        return temp

def _multiproduce(member, d, b, target, id):
    length = len(d)
    for i, data in enumerate(d):
        temp = member._produce(data, b)
        target[id * length + i] = temp

def integer(b):
    return int.from_bytes(b, byteorder="big")
