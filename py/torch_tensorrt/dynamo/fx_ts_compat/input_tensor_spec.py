from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch

from torch_tensorrt.fx.types import Shape, ShapeRange
from torch_tensorrt.fx.utils import get_dynamic_dims
from torch_tensorrt._Input import Input


class InputTensorSpec(NamedTuple):
    """
    This class contains the information of a input tensor.

    shape: shape of the tensor.

    dtype: dtyep of the tensor.

    device: device of the tensor. This is only used to generate inputs to the given model
        in order to run shape prop. For TensorRT engine, inputs have to be on cuda device.

    shape_ranges: If dynamic shape is needed (shape has dimensions of -1), then this field
        has to be provided (default is empty list). Every shape_range is a tuple of three
        tuples ((min_input_shape), (optimized_input_shape), (max_input_shape)). Each shape_range
        is used to populate a TensorRT optimization profile.
        e.g. If the input shape varies from (1, 224) to (100, 224) and we want to optimize
        for (25, 224) because it's the most common input shape, then we set shape_ranges to
        ((1, 224), (25, 225), (100, 224)).

    has_batch_dim: Whether the shape includes batch dimension. Batch dimension has to be provided
        if the engine want to run with dynamic shape.
    """

    shape: Shape
    dtype: torch.dtype
    device: torch.device = torch.device("cpu")
    shape_ranges: List[ShapeRange] = []
    has_batch_dim: bool = True

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "InputTensorSpec":
        """
        Produce an InputTenosrSpec named tuple which contains the
        information of the given PyTorch tensor.

        Args:
            tensor (torch.Tensor): A PyTorch tensor.

        Returns:
            An InputTensorSpec named tuple.
        """
        return cls(tensor.shape, tensor.dtype, tensor.device)

    @classmethod
    def from_tensors(cls, tensors: Sequence[torch.Tensor]) -> List["InputTensorSpec"]:
        """
        Produce a list of InputTenosrSpec named tuples which contain
        the information of all the given PyTorch tensors.

        Args:
            tensors (Iterable[torch.Tensor]): A list of PyTorch tensors.

        Returns:
            A list of InputTensorSpec named tuples.
        """
        assert isinstance(tensors, (list, tuple))
        return [cls.from_tensor(t) for t in tensors]

    @classmethod
    def from_input(cls, input_obj: Input) -> "InputTensorSpec":
        """
        Produce a list of InputTenosrSpec named tuples which contain
        the information of all the given PyTorch tensors.

        Args:
            tensors (Iterable[torch.Tensor]): A list of PyTorch tensors.

        Returns:
            A list of InputTensorSpec named tuples.
        """
        assert isinstance(input_obj, Input)
        input_spec = None
        dtype = input_obj.torch_dtype
        if isinstance(input_obj.shape, dict):
            min_shape = input_obj.shape["min_shape"]
            opt_shape = input_obj.shape["opt_shape"]
            max_shape = input_obj.shape["max_shape"]
            dyn_shape = []
            for min, opt, max in zip(min_shape, opt_shape, max_shape):
                if min == opt == max:
                    dyn_shape.append(min)
                else:
                    dyn_shape.append(-1)
            return cls(
                shape=dyn_shape,
                dtype=dtype,
                shape_ranges=[(min_shape, opt_shape, max_shape)],
            )
        else:
            shape = input_obj.shape
            return cls(shape=shape, dtype=dtype)

    @classmethod
    def from_tensors_with_dynamic_batch_size(
        cls,
        tensors: Sequence[torch.Tensor],
        batch_size_range: Tuple[int, int, int],
        opt_profile_replica: int = 1,
        batch_dims: Optional[List[int]] = None,
    ) -> List["InputTensorSpec"]:
        """
        Produce a list of InputTenosrSpec named tuples which would contain
        the information of all the given PyTorch tensors. The produced input
        tensor specs will treat all tensors' first dimension as batch dimension
        and mark them as dynmaic.

        Args:
            tensors (Sequence[torch.Tensor]): A list of PyTorch tensors.
            batch_size_range (Tuple[int, int, int]): The first integer indicates
                the smallest batch size allowed. The second integer indiceates
                the batch size that we'll optimize for. The third integer indicates
                the largest batch size allowed.
            opt_profile_replica (int): If dynamic shape is enabled, each execution
                context requires a different optimization profile. This arg determines
                how many optimization profile replicas we want to produce.
            batch_dims (Optional[List[int]]): The batch dim might not be the leading dim
                and allow user to specify the batch dims using this arg. Default we treat
                dim 0 as the batch dim.

        Returns:
            A list of InputTensorSpec named tuples with dynamic ranges.
        """
        if batch_dims is None:
            batch_dims = [0] * len(tensors)

        input_specs = []
        batch_size = tensors[0].size(batch_dims[0])

        for i, tensor in enumerate(tensors):
            batch_dim = batch_dims[i]
            assert batch_size == tensor.size(
                batch_dim
            ), f"The {i}th tensor (shape: {tensor.shape}) doesn't have the correct batch size: {batch_size}."
            shape = list(tensor.shape)
            shape[batch_dim] = -1
            shape_ranges: List[ShapeRange] = [
                tuple(
                    tuple(shape[:batch_dim] + [bs] + shape[batch_dim + 1 :])
                    for bs in batch_size_range
                )
            ] * opt_profile_replica
            input_specs.append(
                cls(tuple(shape), tensor.dtype, tensor.device, shape_ranges)
            )

        return input_specs

    def to_random_tensor(self, id=1):
        shape = tuple(self.shape)
        if len(get_dynamic_dims(shape)):
            # id=0 -> min shape
            # id=1 -> optimal shape
            # id=2 -> max shape
            shape = tuple(self.shape_ranges[0][id])
        elif not self.has_batch_dim:
            shape = (1,) + tuple(shape)

        return torch.randn(shape).to(dtype=self.dtype, device=self.device)

    @staticmethod
    def create_inputs_from_specs(input_specs: Iterable["InputTensorSpec"]):
        return [spec.to_random_tensor() for spec in input_specs]

    @staticmethod
    def create_inputs_from_max_specs(input_specs: Iterable["InputTensorSpec"]):
        return [spec.to_random_tensor(2) for spec in input_specs]
