from nn_scratch import Conv2D, ReLU, Sigmoid, AvgPool2D, Upsample2D, Sequential


class ConvAutoencoder:
    """Small convolutional autoencoder used for all experiments."""

    def __init__(self):
        # Encoder downsamples twice, decoder upsamples back to the input size.
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
        """Run the image batch through the autoencoder."""
        return self.net.forward(x)

    def backward(self, grad):
        """Backpropagate reconstruction-loss gradients."""
        return self.net.backward(grad)

    def params(self):
        """Return trainable parameters and their gradients."""
        return self.net.params()
