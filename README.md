# Einop

_One op to rule them all_

Einop is a very thin wrapper around [einops](https://github.com/arogozhnikov/einops) that combines `rearrange`, `reduce`, and `repeat` into a single `einop` function. This library is a port of [arogozhnikov/einops#91](https://github.com/arogozhnikov/einops/pull/91) by [Miles Cranmer](https://github.com/MilesCranmer) into a separate library, if at some point that PR is merged use `einop` directly from einops instead.

## Installation
```
pip install einop
```
## Usage
```python
import numpy as np
from einop import einop

x = np.random.uniform(size=(10, 20))
y = einop(x, "height width -> batch width height", batch=32)

assert y.shape == (32, 20, 10)
```

#### Rearrange
```python
x = np.random.randn(100, 5, 3)

einop(x, 'i j k -> k i j').shape
>>> (3, 100, 5)
```

#### Reduction
```python
x = np.random.randn(100, 5, 3)

einop(x, 'i j k -> i j', reduction='sum').shape
>>> (100, 5)
```

#### Repeat
```python
x = np.random.randn(100, 5, 3)

einop(x, 'i j k -> i j k l', l=10).shape
>>> (100, 5, 3, 10)
```
