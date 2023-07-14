import logging
from typing import Sequence
import torch
from functools import partial
import torch._dynamo as td

from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.lowering._decompositions import (
    get_decompositions,
)
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import (
    pre_aot_substitutions,
)
from torch_tensorrt.dynamo.lowering._partition import (
    partition,
    get_submod_inputs,
)
from torch_tensorrt.dynamo.utils import parse_dynamo_kwargs
from torch_tensorrt.dynamo.conversion import (
    convert_module,
    repair_long_or_double_inputs,
)

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler


logger = logging.getLogger(__name__)


@td.register_backend(name="torch_tensorrt")
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs
):
    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    return DEFAULT_BACKEND(gm, sample_inputs, **kwargs)


@td.register_backend(name="aot_torch_tensorrt_aten")
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs
):
    settings = parse_dynamo_kwargs(kwargs)

    custom_backend = partial(
        _pretraced_backend,
        settings=settings,
    )

    # Perform Pre-AOT Lowering for Module-Level Replacement
    gm = pre_aot_substitutions(gm)

    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(),
    )


def _pretraced_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    """Helper function to manage translation of traced FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    try:
        logger.debug("Post-AOT Autograd graph:\n" + str(gm.graph))

        trt_compiled = _compile_module(
            gm,
            sample_inputs,
            settings=settings,
        )
        return trt_compiled
    except:
        if not settings.pass_through_build_failures:
            logger.warning(
                "TRT conversion failed on the subgraph. See trace above. "
                + "Returning GraphModule forward instead.",
                exc_info=True,
            )
            return gm.forward
        else:
            raise AssertionError(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )


def _compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile a traced FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    # Partition module into components that can be TRT-accelerated
    partitioned_module = partition(
        gm,
        verbose=settings.debug,
        min_block_size=settings.min_block_size,
        torch_executed_ops=settings.torch_executed_ops,
    )

    # Store TRT replicas of Torch subgraphs
    trt_modules = {}

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)

        # Get submodule inputs
        submodule_inputs = get_submod_inputs(
            partitioned_module, submodule, sample_inputs
        )

        # Ensure all submodule inputs do not require a gradient
        for param in submodule_inputs:
            param.requires_grad = False

        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            submodule_inputs = repair_long_or_double_inputs(
                partitioned_module, submodule, submodule_inputs, name
            )

        # Create TRT Module from submodule
        trt_mod = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )

        trt_modules[name] = trt_mod

    # Replace all FX Modules with TRT Modules
    for name, trt_mod in trt_modules.items():
        setattr(partitioned_module, name, trt_mod)

    return partitioned_module
