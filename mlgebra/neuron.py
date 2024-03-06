from exception import *

class Neuron:

    def __init__(self, w_count: int,
                 initialization: str = "uxavier",
                 decimal: bool = False,
                 a: Union[int, float, Decimal] = -1,
                 b: Union[int, float, Decimal] = 1,
                 template: bool = False):

        self.next_weights = Vector.zero(w_count, decimal)

        if template:
            self.weights = Vector()
        else:
            if initialization == "zero":
                self.weights = Vector.zero(w_count, decimal)
            elif initialization == "one":
                self.weights = Vector.one(w_count, decimal)
            elif initialization == "uniform" or initialization == "flat":
                self.weights = Vector.randVfloat(w_count, a, b, decimal)
            elif initialization == "naive":
                self.weights = Vector.randVgauss(w_count, a, b, decimal)
            elif initialization == "uxavier":
                # a will be the neuron count at the current layer here.
                # w_count is the "input_shape"
                limiter = sqrt(6 / (a + w_count))
                self.weights = Vector.randVfloat(w_count, -limiter, limiter, decimal)
            elif initialization == "nxavier" or initialization == "xavier":
                sigma = sqrt(2 / (a + w_count))
                self.weights = Vector.randVgauss(w_count, 0, sigma, decimal)
            elif initialization == "he":
                sigma = 2 / sqrt(a)
                self.weights = Vector.randVgauss(w_count, 0, sigma, decimal)
            else:
                self.weights = Vector(*[a for k in range(w_count)])

