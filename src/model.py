from nn_scratch import Conv2D, ReLU, Sigmoid, AvgPool2D, Upsample2D, Sequential


class ConvAutoencoder:
    def __init__(self):
        self.net = Sequential([
            Conv2D(1, 8, k=3, padding=1),
            ReLU(),
            AvgPool2D(2),

            Conv2D(8, 16, k=3, padding=1),
            ReLU(),
            AvgPool2D(2),

            Upsample2D(2),
            Conv2D(16, 8, k=3, padding=1),
            ReLU(),

            Upsample2D(2),
            Conv2D(8, 1, k=3, padding=1),
            Sigmoid()
        ])

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, grad):
        return self.net.backward(grad)

    def params(self):
        return self.net.params()