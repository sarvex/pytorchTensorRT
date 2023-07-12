"""
.. _dynamo_module_level_acceleration:

Dynamo Module Level Acceleration Tutorial
=========================

This interactive script is intended as an overview of the process by which module-level acceleration for `torch_tensorrt.dynamo.compile` works, and how it can be used to accelerate built-in or custom `torch.nn` modules by excluding them from AOT tracing. This script shows the process for `torch.nn.MaxPool1d`"""

# %%
# 1. The Placeholder
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Specify the schema and namespace of the operator, as well as a placeholder function
# representing the schema. The schema should be in torch JIT syntax, indicating input and output
# types. The namespace, such as tensorrt, will cause the op to be registered as `torch.ops.tensorrt.your_op`
# Then, create a placeholder function with no operations, but having the same schema and naming as that
# used in the decorator

# %%

from torch._custom_op.impl import custom_op


@custom_op(
    qualname="tensorrt::maxpool1d",
    manual_schema="(Tensor x, int[1] kernel_size, int[1] stride, int[1] padding, int[1] dilation, bool ceil_mode) -> Tensor",
)
def maxpool1d(x, kernel_size, stride, padding, dilation, ceil_mode):
    # Defines operator schema, name, namespace, and function header
    ...


# %%
# 2. The Generic Implementation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Define the default implementation of the operator in torch syntax. This is used for autograd
# and other tracing functionality. Generally, the `torch.nn.functional` analog of the operator to replace
# is desirable. If the operator to replace is a custom module you've written, then add its Torch
# implementation here. Note that the function header to the generic function can have specific arguments
# as in the above placeholder

# %%
import torch


@maxpool1d.impl("cpu")
@maxpool1d.impl("cuda")
@maxpool1d.impl_abstract()
def maxpool1d_generic(
    *args,
    **kwargs,
):
    # Defines an implementation for AOT Autograd to use for shape analysis/propagation
    return torch.nn.functional.max_pool1d(
        *args,
        **kwargs,
    )


# %%
# 3. The Module Substitution Function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Define a function which can intercept a node of the kind to be replaced, extract
# the relevant data from that node/submodule, and then re-package the information
# for use by an accelerated implementation (to be implemented in step 4). This function
# should use the operator defined in step 1 (for example `torch.ops.tensorrt.maxpool1d`).
# It should refactor the args and kwargs as is needed by the accelerated implementation.

# %%

from torch_tensorrt.dynamo.backend.lowering import register_substitution


@register_substitution(torch.nn.MaxPool1d, torch.ops.tensorrt.maxpool1d)
def maxpool1d_insertion_fn(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    submodule: torch.nn.Module,
) -> torch.fx.Node:
    # Defines insertion function for new node
    new_node = gm.graph.call_function(
        torch.ops.tensorrt.maxpool1d,
        args=node.args,
        kwargs={
            "kernel_size": submodule.kernel_size,
            "stride": submodule.stride,
            "padding": submodule.padding,
            "dilation": submodule.dilation,
            "ceil_mode": submodule.ceil_mode,
        },
    )

    return new_node


# %%
# If the submodule has weights or other Tensor fields which the accelerated implementation
# needs, the function should insert the necessary nodes to access those weights. For example,
# if the weight Tensor of a submodule is needed, one could write::
#
#
#       weights = gm.graph.get_attr(n.target + ".weight", torch.Tensor)
#       bias = gm.graph.get_attr(n.target + ".bias", torch.Tensor)
#
#       ...
#
#       kwargs={"weight": weights,
#               "bias": bias,
#               ...
#              }

# %%
# 4. The Accelerated Implementation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Define an accelerated implementation of the operator, and register it as necessary.
# This accelerated implementation should consume the args/kwargs specified in step 3.
# One should expect that torch.compile will compress all kwargs into the args field in
# the order specified in the schema written in step 1.

# %%

from typing import Dict, Tuple
from torch.fx.node import Argument, Target
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters import acc_ops_converters


@tensorrt_converter(torch.ops.tensorrt.maxpool1d.default)
def tensorrt_maxpool1d(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    # Defines converter replacing the default operator for this function
    kwargs_new = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": args[2],
        "padding": args[3],
        "dilation": args[4],
        "ceil_mode": False if len(args) < 6 else args[5],
    }

    return acc_ops_converters.acc_ops_max_pool1d(
        network, target, None, kwargs_new, name
    )


# %%
# 5. Add Imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add your accelerated module file to the `__init__.py` in the
# `py/torch_tensorrt/dynamo/backend/lowering/substitutions` directory, to ensure
# all registrations are run. For instance, if the new module file is called `new_mod.py`,
# one should add `from .new_mod import *` to the `__init__.py`
