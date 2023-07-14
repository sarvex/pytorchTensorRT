import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Union
from enum import Enum, auto

from torch.fx.node import Target, Node, _get_qualified_name
from torch_tensorrt.fx.converter_registry import CONVERTERS


logger = logging.getLogger(__name__)


class ConverterPriority(Enum):
    """Enum to set a converter's priority in the registry"""

    STANDARD = auto()
    HIGH = auto()


@dataclass(frozen=True)
class ConverterSupport:
    """Class representing a converter implementation and support function

    Args:
        converter_implementation: Function which converts said node to a TRT equivalent
        capability_validator: Function which takes in a Node and returns a bool indicating
            whether that node can be supported by its companion converter. Note that
            this function must not modify the node or its graph
    """

    converter_implementation: Callable
    capability_validator: Callable[[Node], bool] = field(default=lambda node: True)


# Dictionary representing Dynamo aten-only converters
# Each converter maps to a sequence of at least one ConverterSupport object(s)
DYNAMO_ATEN_CONVERTERS: Dict[Target, Sequence[ConverterSupport]] = {}


def dynamo_tensorrt_converter(
    key: Target,
    enabled: bool = True,
    capability_validator: Optional[Callable[[Node], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
) -> Callable[[Any], Any]:
    """Decorator for Dynamo TensorRT Converter

    Registers the decorated function in the DYNAMO_ATEN_CONVERTERS registry

    Args:
        key: Node target for which the converter is implemented for
            (for example, torch.ops.add.Tensor)
        enabled: Whether the converter should be enabled/cached or not
        capability_validator: Function which evaluates whether a node is valid for conversion
            by the decorated converter. See ConverterSupport for more details.
            Defaults to None, implying the capability_validator function is always true -
            this means all nodes of "key" kind can be supported by this converter
        priority: Converter's level of priority relative to other converters with the
            same target
    Returns:
        The converter being decorated
    """

    def register_converter(converter):
        """Helper function to register the converter, then return it"""
        assert callable(converter), "Converter function must be callable"

        # If no capability_validator function is specified, use the default function - always return true
        if capability_validator is None:
            converter_support = ConverterSupport(converter_implementation=converter)
        else:
            assert callable(
                capability_validator
            ), "Argument checking function must be callable"
            converter_support = ConverterSupport(
                converter_implementation=converter,
                capability_validator=capability_validator,
            )

        # If a converter for this operator already exists, append the new converter to the list
        # Otherwise, start a new list
        if key in DYNAMO_ATEN_CONVERTERS:
            # High priority converters are inserted at the front of the list,
            # so they can be checked first by the registry
            if priority is ConverterPriority.HIGH:
                DYNAMO_ATEN_CONVERTERS[key].insert(0, converter_support)
            else:
                DYNAMO_ATEN_CONVERTERS[key].append(converter_support)
        else:
            DYNAMO_ATEN_CONVERTERS[key] = [converter_support]

        logger.debug(
            f"Converter for {key} added to Dynamo ATen Converter Registry with priority: {priority}"
        )

        return converter

    def disable_converter(converter):
        return converter

    # Select whether to cache/enable the converter
    if enabled:
        return register_converter
    else:
        return disable_converter


class ConverterRegistry:
    """Registry for storing multiple converter dictionaries

    Capable of storing dictionaries with the following signature:
    Dict[Target, Union[Callable, Sequence[ConverterSupport]]]

    Also able to validate converter implementations against user-provided
    argument-checking functions

    Args:
        registries: List of dictionaries representing converter registries.
            The order of the provided dictionaries is the order in which they
            will be traversed. This is only significant when using non-validated
            methods.
    """

    def __init__(
        self,
        registries: Sequence[Dict[Target, Union[Callable, Sequence[ConverterSupport]]]],
        registry_names: Optional[Sequence[str]] = None,
    ):
        # Copy reference to each dictionary object into attribute list
        self.registries = [registry for registry in registries]

        if registry_names is not None:
            assert len(self.registries) == len(registry_names)
            self.registry_names = [name for name in registry_names]
        else:
            self.registry_names = [
                f"Registry {i + 1}" for i in range(len(self.registries))
            ]

        self.validate_invariants()

    def validate_invariants(self):
        """Validates the invariants required of the dictionaries in the registries

        Raises AssertionError if any invariants have been violated
        """
        # All registries must be dictionaries
        assert all(isinstance(elt, dict) for elt in self.registries)

        # Every dictionary in the registry must have one of two signatures:
        # Dict[Target, Callable] or Dict[Target, Sequence[ConverterSupport]]
        # Where, for the latter, the sequence must be non-empty
        for registry in self.registries:
            for converters in registry.values():
                if isinstance(converters, (list, tuple)):
                    assert (
                        all(isinstance(c, ConverterSupport) for c in converters)
                        and len(converters) > 0
                    )
                else:
                    assert callable(converters), "Converter function must be callable"

    def __getitem_without_validation__(self, key: Target):
        """Get the first-found converter in any registry

        Searches all registries in order and returns the first converter encountered
        """
        if isinstance(key, Node):
            raise KeyError(
                "Unvalidated accesses to the Converter registry can only be "
                + "made with node targets. Try accessing the registry with node.target"
            )

        self.validate_invariants()

        # Iterate over all registries and return the first converter found
        for registry in self.registries:
            if key in registry:
                converters = registry[key]

                if isinstance(converters, (list, tuple)):
                    return converters[0].converter_implementation
                else:
                    return converters

        raise KeyError(f"None of the converter registries have an entry for {key}")

    def __getitem__(self, node: Node):
        """Get the first-found validated converter in any registry

        Searches all registries in order and returns the first converter
        which passes validation on the input node
        """
        if not isinstance(node, Node):
            raise KeyError(
                "Validated accesses to the Converter registry can only be "
                + "made with node inputs. Try accessing the registry with a node "
                + "or use get_unvalidated to access without node validation."
            )

        self.validate_invariants()
        key = node.target

        # Iterate over all registries, validating the converter on the input node
        # If no capability_validator function is found, assume full coverage
        for registry in self.registries:
            if key in registry:
                converters = registry[key]

                if isinstance(converters, (list, tuple)):
                    for candidate in converters:
                        if candidate.capability_validator(node):
                            return candidate.converter_implementation
                else:
                    return converters

        raise KeyError(
            f"None of the converter registries have a validated entry for {key}, with node {node}"
        )

    def keys(self):
        """Get all unique targets across all dictionaries"""
        return self.unique_targets()

    def get_unvalidated(self, key: Target, value=None):
        """Get unvalidated converter for input target with a default return"""
        try:
            return self.__getitem_without_validation__(key)
        except KeyError:
            return value

    def get(self, node: Node, value=None):
        """Get validated converter for input node with a default return"""
        try:
            return self.__getitem__(node)
        except KeyError:
            return value

    def __contains__(self, key: Union[Target, Node]):
        """Check whether a converter for an input node or target exists"""
        try:
            # Attempt to access the item in the registry
            if isinstance(key, Node):
                self.__getitem__(key)
            else:
                self.__getitem_without_validation__(key)

            return True
        except KeyError:
            return False

    def get_all_converters_with_target(
        self, key: Target, return_registry_info: bool = False
    ):
        """Get all converters across all registries for the target

        Returns a list of all converterts having the specified target
        """
        self.validate_invariants()
        converters_with_target = []

        # Store count of number of registered converters per registry
        if return_registry_info:
            registry_data = {name: 0 for name in self.registry_names}

        for index, registry in enumerate(self.registries):
            if key in registry:
                converters = registry[key]

                if isinstance(converters, (list, tuple)):
                    converters_with_target.extend(
                        [c.converter_implementation for c in converters]
                    )
                    # Add converter count to registry name storage
                    if return_registry_info:
                        registry_data[self.registry_names[index]] += len(converters)
                else:
                    converters_with_target.append(converters)
                    # Add converter count to registry name storage
                    if return_registry_info:
                        registry_data[self.registry_names[index]] += 1

        if return_registry_info:
            return converters_with_target, registry_data
        else:
            return converters_with_target

    def __setitem__(self, key, value):
        raise AssertionError(
            f"Do not set registry members directly through the ConverterRegistry object. "
            + f"Attempted to set {key}: {value} via direct assignment to ConverterRegistry."
        )

    def __delitem__(self, key):
        raise AssertionError(
            f"Do not delete registry members directly through the ConverterRegistry object. "
            + f"Attempted to delete {key} via direct del on ConverterRegistry."
        )

    def __len__(self):
        """Returns the sum of lengths of all registries stored"""
        return sum(len(registry) for registry in self.registries)

    def unique_targets(self):
        """Returns the set of unique converter targets stored across all registries"""
        return set.union(*[set(registry.keys()) for registry in self.registries])

    def qualified_name_or_str(self, target: Target) -> str:
        """Returns string representation of an FX Node target"""
        if isinstance(target, str):
            return target
        else:
            return _get_qualified_name(target)

    def display_all_available_converters(self) -> str:
        """Returns a string with all converters and their source, separated by newlines"""
        available_converters = "Available converters in ATen registries with counts:\n"

        for target in sorted(
            self.unique_targets(), key=lambda target: self.qualified_name_or_str(target)
        ):
            _, registry_data = self.get_all_converters_with_target(
                target, return_registry_info=True
            )
            available_converters += f"Node: {self.qualified_name_or_str(target)} - Registry Presence Counts: {registry_data}\n"

        return available_converters


# Initialize dynamo converter registry with the FX and Dynamo aten registries
# Note the Dynamo registry is listed first, for precedence
DYNAMO_CONVERTERS: ConverterRegistry = ConverterRegistry(
    [DYNAMO_ATEN_CONVERTERS, CONVERTERS],
    ["Dynamo ATen Converters Registry", "FX ATen Converters Registry"],
)
