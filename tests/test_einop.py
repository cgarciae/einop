import numpy as np

from einop import einop

# TODO: make this a fixture
x = np.arange(10 * 20 * 30 * 40).reshape([10, 20, 30, 40])


class TestEinop:
    def test_generic(self):
        # concatenate
        tensors = list(
            x + 0
        )  # 0 is needed https://github.com/tensorflow/tensorflow/issues/23185
        tensors = einop(tensors, "b c h w -> h (b w) c")
        assert tensors.shape == (30, 10 * 40, 20)
        return tensors

    def test_maxpooling(self):
        # max-pooling
        y = einop(x, "b c (h h1) (w w1) -> b c h w", reduction="max", h1=2, w1=2)
        assert y.shape == (10, 20, 30 // 2, 40 // 2)
        return y

    def test_squeeze(self):
        # squeeze - unsqueeze
        y = einop(x, "b c h w -> b c () ()", reduction="max")
        assert y.shape == (10, 20, 1, 1)
        y = einop(y, "b c () () -> c b")
        assert y.shape == (20, 10)
        return y

    def test_example(self):
        x = np.random.uniform(size=(10, 20))
        y = einop(x, "height width -> batch width height", batch=32)

        assert y.shape == (32, 20, 10)
