import flax.linen as nn
from typing import Sequence


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            # print(layer)
            if isinstance(layer, nn.Module):
                x = layer(x, *args, **kwargs)
            else:
                x = layer(x)
        return x
