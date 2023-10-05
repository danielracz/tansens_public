from typing import Callable

from flax import linen as nn
from flax.linen.initializers import lecun_normal, zeros
from jax.nn.initializers import Initializer
from nptyping import Float32, Int32, NDArray, Shape


class FCLayer(nn.Module):
    features: int
    dropout_rate: float | None = None
    activation_fn: Callable | None = None
    kernel_init: Initializer = lecun_normal()
    bias_init: Initializer = zeros

    @nn.compact
    def __call__(self, x, training: bool | None = None):
        x = nn.Dense(self.features,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)

        if self.dropout_rate is not None:
            if training is None:
                raise ValueError("training is None when using dropout")
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x


class FCNet(nn.Module):

    W: int = 10_000
    # TODO: W = 500, W = 784, W = 1200, W = 2*784, W = 3000, W = 30_000
    kernel_init: Initializer = lecun_normal()
    bias_init: Initializer = zeros

    @nn.compact
    def __call__(self, x: NDArray[Shape["N, H, W, C"], Float32],
                 training: bool | None = None
                 ) -> NDArray[Shape["N, 10"], Int32]:

        x = x.reshape((x.shape[0], -1))

        x = FCLayer(features=self.W,
                    dropout_rate=0.25,
                    activation_fn=nn.relu,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(x, training=training)

        x = FCLayer(features=self.W,
                    dropout_rate=0.5,
                    activation_fn=nn.relu,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(x, training=training)

        x = FCLayer(features=self.W,
                    dropout_rate=0.5,
                    activation_fn=nn.relu,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(x, training=training)

        self.sow('intermediates', 'h', x)

        x = FCLayer(features=10)(x)

        return x


class FCNet2(nn.Module):

    Ws: tuple[int]
    num_classes: int = 10
    kernel_init: Initializer = lecun_normal()
    bias_init: Initializer = zeros

    @nn.compact
    def __call__(self, x: NDArray[Shape["N, H, W, C"], Float32],
                 training: bool | None = None
                 ) -> NDArray[Shape["N, 10"], Int32]:

        x = x.reshape((x.shape[0], -1))

        for W in self.Ws:

            x = FCLayer(features=W,
                        activation_fn=nn.relu,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init)(x, training=training)

        self.sow('intermediates', 'h', x)

        x = FCLayer(features=self.num_classes)(x)

        return x


class FCNetBin(nn.Module):
    
    Ws: tuple[int]
    kernel_init: Initializer = lecun_normal()
    bias_init: Initializer = zeros
    
    @nn.compact
    def __call__(self, x: NDArray[Shape["N, H, W, C"], Float32],
                 training: bool | None = None
                 ) -> NDArray[Shape["N"], Float32]:
        
        x = x.reshape((x.shape[0], -1))
        for W in self.Ws:
            x = FCLayer(features=W,
                        activation_fn=nn.relu,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init)(x, training=training)
            
        self.sow('intermediates', 'h', x)
        x = FCLayer(features=1)(x)
        return x.squeeze()
            



class ConvNet(nn.Module):

    kernel_init: Initializer = lecun_normal()
    bias_init: Initializer = zeros

    @nn.compact
    def __call__(self, x: NDArray[Shape["N, H, W, C"], Float32],
                 training: bool
                 ) -> NDArray[Shape["N, 10"], Int32]:

        x = nn.Conv(features=64, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        x = nn.Conv(features=64, kernel_size=(3, 3), padding='VALID',
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='VALID',
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=64, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

        return x


class Dense(nn.Module):
    @nn.compact
    def __call__(self, x, training):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x