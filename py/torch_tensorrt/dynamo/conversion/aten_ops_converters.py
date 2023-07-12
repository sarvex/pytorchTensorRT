import logging
from typing import Dict, Sequence, Tuple, Union
import torch
import operator
from torch_tensorrt.fx.converters import acc_ops_converters
from ..converter_registry import dynamo_tensorrt_converter
from torch.fx.node import Argument, Target, Node

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.dynamo.conversion import SourceIR, impl

_LOGGER: logging.Logger = logging.getLogger(__name__)


def or_none(args, i):
    return args[i] if len(args) > i else None


@dynamo_tensorrt_converter(torch.ops.aten.batch_norm)
def aten_ops_batch_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
    )


@dynamo_tensorrt_converter(torch.ops.aten.div.default)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor_mode)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor)
def aten_ops_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    rounding_mode = kwargs.get("rounding_mode")
    if rounding_mode is None:
        return acc_ops_converters.acc_ops_div(network, target, None, kwargs_new, name)
    elif rounding_mode == "floor":
        return acc_ops_converters.acc_ops_floor_div(
            network, target, None, kwargs_new, name
        )
    elif rounding_mode == "trunc":
        return impl.elementwise.trunc_div(
            network, target, SourceIR.ATEN, name, args[0], args[1]
        )
    else:
        raise RuntimeError(
            f"Target {target} does not support rounding mode {rounding_mode}"
        )


@dynamo_tensorrt_converter(torch.ops.aten.fmod.Scalar)
@dynamo_tensorrt_converter(torch.ops.aten.fmod.Tensor)
def aten_ops_fmod(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.fmod(network, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.gelu.default)
def aten_ops_gelu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.gelu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.matmul)
@dynamo_tensorrt_converter(torch.ops.aten.mm.default)
def aten_ops_matmul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.matmul.matrix_multiply(
        network, target, SourceIR.ATEN, name, args[0], args[1]
    )


@dynamo_tensorrt_converter(torch.ops.aten.layer_norm.default)
def aten_ops_layernorm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.layer_norm(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
    )


@dynamo_tensorrt_converter(torch.ops.aten.relu.default)
def aten_ops_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    return impl.activation.relu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.rsqrt.default)
def aten_ops_rsqrt(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    return impl.elementwise.rsqrt(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dim)
@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dims)
def aten_ops_squeeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.squeeze.squeeze(network, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.unsqueeze.default)
def aten_ops_unsqueeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unsqueeze.unsqueeze(
        network, target, SourceIR.ATEN, name, input_t=args[0], dim=args[1]
    )


@dynamo_tensorrt_converter(torch.ops.aten._softmax.default)
def aten_ops_softmax(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.softmax(
        network, target, SourceIR.ATEN, name, args[0], args[1]
    )


@dynamo_tensorrt_converter(torch.ops.aten.where.self)
def aten_ops_where(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.condition.where(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[1],
        args[2],
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.clamp.default)
def aten_ops_clamp(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.clamp(
        network,
        target,
        SourceIR.ACC,
        name,
        input_val=args[0],
        min_val=or_none(args, 1),
        max_val=or_none(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.select.int)
def aten_ops_select(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.select(
        network, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(torch.ops.aten.slice.Tensor)
def aten_ops_slice(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.slice_op(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
    )


def to_copy_dtype_validator(to_copy_node: Node):
    allowed_casts = {torch.float, torch.int32, torch.bool, torch.int8, torch.float16}

    # Validate input node has convertible kwargs
    if "dtype" in to_copy_node.kwargs:
        if to_copy_node.kwargs["dtype"] in allowed_casts:
            return True
        else:
            _LOGGER.debug(
                f"_to_copy converter rejected node {to_copy_node} with dtype {to_copy_node.kwargs['dtype']}"
            )
            return False
    else:
        _LOGGER.debug(
            f"_to_copy converter rejected node {to_copy_node} with kwargs {to_copy_node.kwargs}"
        )
        return False


@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default, capability_validator=to_copy_dtype_validator
)
def aten_ops_to_copy_dtype(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.cast.to_copy(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs["dtype"],
    )


@dynamo_tensorrt_converter(operator.getitem)
def operator_getitem(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.evaluators.getitem(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.clone.default)
def aten_ops_clone(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.evaluators.clone(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )
