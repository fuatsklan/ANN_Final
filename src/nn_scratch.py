import numpy as np


class Conv2D:
    """Naive 2D convolution layer with manual gradients."""

    def __init__(self, in_ch, out_ch, k=3, padding=1):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.padding = padding

        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = np.random.randn(out_ch, in_ch, k, k) * scale
        self.b = np.zeros((out_ch, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        p = self.padding
        k = self.k

        xpad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        self.xpad = xpad

        out = np.zeros((B, self.out_ch, H, W))

        for i in range(H):
            for j in range(W):
                region = xpad[:, :, i:i+k, j:j+k]
                out[:, :, i, j] = np.tensordot(
                    region, self.W, axes=([1, 2, 3], [1, 2, 3])
                ) + self.b[:, 0]

        return out

    def backward(self, dout):
        B, C, H, W = self.x.shape
        p = self.padding
        k = self.k

        dxpad = np.zeros_like(self.xpad)
        self.dW.fill(0)
        self.db = np.sum(dout, axis=(0, 2, 3)).reshape(self.out_ch, 1)

        for i in range(H):
            for j in range(W):
                region = self.xpad[:, :, i:i+k, j:j+k]

                for oc in range(self.out_ch):
                    self.dW[oc] += np.sum(
                        region * dout[:, oc:oc+1, i:i+1, j:j+1],
                        axis=0
                    )

                for b in range(B):
                    dxpad[b, :, i:i+k, j:j+k] += np.sum(
                        self.W * dout[b, :, i, j].reshape(-1, 1, 1, 1),
                        axis=0
                    )

        if p > 0:
            return dxpad[:, :, p:-p, p:-p]
        return dxpad

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """ReLU activation."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params(self):
        return []


class Sigmoid:
    """Sigmoid activation used to keep reconstructed pixels in [0, 1]."""

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-np.clip(x, -30, 30)))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

    def params(self):
        return []


class AvgPool2D:
    """Average pooling for spatial downsampling."""

    def __init__(self, k=2):
        self.k = k

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        k = self.k

        out = np.zeros((B, C, H // k, W // k))

        for i in range(H // k):
            for j in range(W // k):
                region = x[:, :, i*k:(i+1)*k, j*k:(j+1)*k]
                out[:, :, i, j] = np.mean(region, axis=(2, 3))

        return out

    def backward(self, dout):
        B, C, H, W = self.x.shape
        k = self.k
        dx = np.zeros_like(self.x)

        for i in range(H // k):
            for j in range(W // k):
                dx[:, :, i*k:(i+1)*k, j*k:(j+1)*k] += (
                    dout[:, :, i:i+1, j:j+1] / (k * k)
                )

        return dx

    def params(self):
        return []


class Upsample2D:
    """Nearest-neighbor upsampling for the decoder."""

    def __init__(self, scale=2):
        self.scale = scale

    def forward(self, x):
        self.x = x
        return np.repeat(np.repeat(x, self.scale, axis=2), self.scale, axis=3)

    def backward(self, dout):
        B, C, H, W = self.x.shape
        s = self.scale
        dx = np.zeros_like(self.x)

        for i in range(H):
            for j in range(W):
                dx[:, :, i, j] = np.sum(
                    dout[:, :, i*s:(i+1)*s, j*s:(j+1)*s],
                    axis=(2, 3)
                )

        return dx

    def params(self):
        return []


class Sequential:
    """Tiny sequential container for scratch layers."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self):
        ps = []
        for layer in self.layers:
            ps.extend(layer.params())
        return ps


class MSELoss:
    """Mean squared reconstruction loss."""

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size


class Adam:
    """Adam optimizer over (parameter, gradient) pairs."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p) for p, g in params]
        self.v = [np.zeros_like(p) for p, g in params]

    def step(self):
        self.t += 1

        for i, (p, g) in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
