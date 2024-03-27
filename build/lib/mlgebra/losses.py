from .exception import *

def MSE(y: Union[List[Vector], List[Matrix], List[Tensor]],
        predictions: Union[List[Vector], List[Matrix], List[Tensor]]):
    N = len(y)

    if N != len(predictions):
        raise DimensionError(0)

    s = 0
    for k in range(N):
        s += (y[k] - predictions[k]).dot(y[k] + predictions[k])
    return s / N

# This is just as an alias
def MeanSquarredError(y: Union[List[Vector], List[Matrix], List[Tensor]],
                      predictions: Union[List[Vector], List[Matrix], List[Tensor]]):
    N = len(y)

    if N != len(predictions):
        raise DimensionError(0)

    s = 0
    for k in range(N):
        s += (y[k] - predictions[k]).dot(y[k] + predictions[k])
    return s / N

def SparseCategoricalCrossEntropy(
        y: Union[List[Vector], List[Matrix], List[Tensor]],
        predictions: Union[List[Vector], List[Matrix], List[Tensor]]):
    N = len(y)

    if N != len(predictions):
        raise DimensionError(0)

    s = 0
    for k in range(N):
        s += y[k].dot(predictions[k].map(abs).map(ln))  # abs is used here to compromise floating point errors
    return s

"""

Below gradients are used to start the backpropagation process. After
the starting, errors are calculated on derivatives and chain rule,
not on cross entropy or MSE.

"""
def gradMSE(y: Union[List[Vector], List[Matrix], List[Tensor]],
            predictions: Union[List[Vector], List[Matrix], List[Tensor]]):
    N = len(y)

    if N != len(predictions):
        raise DimensionError(0)

    s = Vector.zero(y[0].dimension)
    for k in range(N):
        s += (y[k] - predictions[k])
    return s * (2 / N)

def gradCrossEntropy(y: Union[List[Vector], List[Matrix], List[Tensor]],
                     predictions: Union[List[Vector], List[Matrix], List[Tensor]]):
    N = len(y)

    if N != len(predictions):
        raise DimensionError(0)

    s = 0
    for k in range(N):
        s += predictions[k] - y[k]
    return s





