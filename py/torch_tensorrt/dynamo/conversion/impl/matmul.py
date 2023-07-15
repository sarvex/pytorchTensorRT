from typing import Optional


import tensorrt as trt
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.utils import (
    unified_dtype_converter,
    Frameworks,
)
from torch_tensorrt.dynamo.conversion import SourceIR
from torch_tensorrt.fx.converters.converter_utils import (
    get_trt_tensor,
    broadcast,
    set_layer_name,
)


def matrix_multiply(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(network, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(
            network,
            other,
            f"{name}_other",
            dtype=unified_dtype_converter(input.dtype, Frameworks.TORCH),
        )

    input_matrix_op = other_matrix_op = trt.MatrixOperation.NONE
    preset_diff = 0

    if len(input.shape) == 1:
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR

    if len(other.shape) == 1:
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR

    input, other = broadcast(
        network, input, other, f"{name}_input", f"{name}_other", preset_diff
    )
    layer = network.add_matrix_multiply(input, input_matrix_op, other, other_matrix_op)
    set_layer_name(layer, target, name)
    return layer.get_output(0)
