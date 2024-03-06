from layer import *
from os.path import exists, isdir
import cProfile

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
            else:
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
                    file.write(":".join(neuron.weights.values) + "\n")
                file.write(":".join(layer.bias.values) + "\n")

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def structure(self, s: List[Layer]):
        self.layers = [layer for layer in s]

    def produce(self, input: Union[Tensor, Matrix, Vector]):
        output = input.reshape(len(input.values) * len(input.values[0]))
        for layer in self.layers:
            output = layer.forward(output)
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
                if k % 600 == 0:
                    print(f"Training {(k / 600):.2f} done.")
                for l in range(batch_size):
                    error += y[k * batch_size + l] - self.produce(x[k * batch_size + l])

                error /= batch_size
                error = self.layers[-1].startBackward(error, self.layers[-2].output, lr)

                for i in range(len(self.layers) - 2, 0, -1):
                    error_sum = []
                    if self.layers[i].name == "dense" and self.layers[i + 1].name == "dense":
                        for j in range(self.layers[i].units):
                            e_sum = 0
                            for m in range(self.layers[i + 1].units):
                                e_sum += self.layers[i + 1].layer[m].weights[j] * error[m]
                            error_sum.append(e_sum)

                        error = Vector(*error_sum)
                        self.layers[i].backward(error, self.layers[i - 1].output, lr)

                for i in range(1, len(self.layers)):
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
                if self.layers[i].name == "dense" and self.layers[i + 1].name == "dense":
                    for j in range(self.layers[i].units):
                        e_sum = 0
                        for m in range(self.layers[i + 1].units):
                            e_sum += self.layers[i + 1].layer[m].weights[j] * error[m]
                        error_sum.append(e_sum)

                    error = Vector(*error_sum)
                    self.layers[i].backward(error, self.layers[i - 1].output, lr)

            for i in range(1, len(self.layers)):
                self.layers[i].update()

            if validation_x:
                logger.info(f"Error at last batch with {self.loss}: {self.evaluate(validation_x, validation_y)}")

    def describe(self):
        print(f"Model: {self.name}")
        for layer in self.layers:
            print(f"-> {layer.name} | {layer.units} | {layer.activation}")


