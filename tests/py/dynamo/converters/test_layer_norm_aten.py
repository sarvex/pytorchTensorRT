import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.test_utils import DispatchTestCase
from torch_tensorrt import Input


class TestLayerNormConverter(DispatchTestCase):
    def test_layer_norm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = torch.nn.LayerNorm([3, 224, 224])

            def forward(self, x):
                return self.ln(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten.layer_norm.default}
        )

    def test_layernorm_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = torch.nn.LayerNorm([3, 224, 224])

            def forward(self, x):
                return self.ln(x)

        input_specs = [
            Input(
                shape=(-1, 3, 224, 224),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 224, 224), (1, 3, 224, 224), (2, 3, 224, 224))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.layer_norm.default}
        )


if __name__ == "__main__":
    run_tests()
