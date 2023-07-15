from torch_tensorrt.dynamo.backend.lowering import partition
from torch.testing._internal.common_utils import run_tests, TestCase
import torch
from copy import deepcopy
from torch_tensorrt.dynamo import compile
from utils import lower_graph_testing
from torch_tensorrt.dynamo.common_utils.test_utils import DECIMALS_OF_AGREEMENT


class TestTRTModuleNextCompilation(TestCase):
    def test_trt_module_next_full_support(self):

        class FullySupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                out = x - y
                out = out + x
                out = 2 * out
                out = out + y
                return torch.mean(out, dim=1)

        fx_graph = torch.fx.symbolic_trace(FullySupportedMultiOp())
        partitioned_graph = partition(deepcopy(fx_graph), min_block_size=3)

        self.assertEquals(
            len(list(partitioned_graph.named_children())),
            1,
            "All operators are supported, there should be one segment",
        )

        inputs = [
            torch.randint(-5, 5, (16, 7), dtype=torch.float).cuda(),
            torch.randint(-5, 5, (16, 7), dtype=torch.float).cuda(),
        ]

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = compile(
            fx_graph,
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            use_python_runtime=False,
            debug=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            "TRT outputs don't match with the original model.",
        )

    def test_trt_module_next_partial_support(self):

        class PartiallySupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                out = x - y
                out = out - 3 * x
                out = out + y
                out = out.to(torch.float)
                out = 2 * out
                return torch.mean(out, dim=-1)

        fx_graph = torch.fx.symbolic_trace(PartiallySupportedMultiOp())
        unexpected_ops = {torch.ops.aten.add.Tensor}

        inputs = [
            torch.randint(-40, 40, (16, 7, 5), dtype=torch.int).cuda(),
            torch.randint(1, 40, (16, 7, 5), dtype=torch.int).cuda(),
        ]

        (unexpected_ops_seen, _, partitioned_graphs,) = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            testing_partitioning=True,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )
        self.assertEquals(
            len(partitioned_graphs),
            1,
            "Without control flow breaks, there should only be a single graph",
        )
        self.assertEquals(
            len(list(partitioned_graphs[0].named_children())),
            2,
            "Certain operators are set to run in Torch, expected 2 segments",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = compile(
            fx_graph,
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            use_python_runtime=False,
            debug=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            "TRT outputs don't match with the original model.",
        )


class TestCompilationOptions(TestCase):
    def test_trt_specific_options(self):

        class SupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                out = x - y
                out = out - 3 * x
                out = out + y
                out = out - y / 5
                out = 2 * out
                return torch.mean(out, dim=-1)

        fx_graph = torch.fx.symbolic_trace(SupportedMultiOp())

        inputs = [
            torch.randint(-40, 40, (16, 7, 5), dtype=torch.float).cuda(),
            torch.randint(1, 40, (16, 7, 5), dtype=torch.float).cuda(),
        ]

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = compile(
            fx_graph,
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=False,
            optimization_level=4,
            version_compatible=True,
            max_aux_streams=5,
            debug=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            "TRT outputs don't match with the original model.",
        )


if __name__ == "__main__":
    run_tests()
