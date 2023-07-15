import torch

# from torch_tensorrt import _enums
import tensorrt as trt
from torch_tensorrt import logging
import warnings

try:
    from torch_tensorrt import _C
except:
    warnings.warn(
        "Unable to import torchscript frontend core and torch-tensorrt runtime. Some dependent features may be unavailable."
    )


class Device(object):
    """
    Defines a device that can be used to specify target devices for engines

    Attributes:
        device_type (torch_tensorrt.DeviceType): Target device type (GPU or DLA). Set implicitly based on if dla_core is specified.
        gpu_id (int): Device ID for target GPU
        dla_core (int): Core ID for target DLA core
        allow_gpu_fallback (bool): Whether falling back to GPU if DLA cannot support an op should be allowed
    """

    device_type = None  #: (torch_tensorrt.DeviceType): Target device type (GPU or DLA). Set implicitly based on if dla_core is specified.
    gpu_id = -1  #: (int) Device ID for target GPU
    dla_core = -1  #: (int) Core ID for target DLA core
    allow_gpu_fallback = False  #: (bool) Whether falling back to GPU if DLA cannot support an op should be allowed

    def __init__(self, *args, **kwargs):
        """__init__ Method for torch_tensorrt.Device

        Device accepts one of a few construction patterns

        Args:
            spec (str): String with device spec e.g. "dla:0" for dla, core_id 0

        Keyword Arguments:
            gpu_id (int): ID of target GPU (will get overrided if dla_core is specified to the GPU managing DLA). If specified, no positional arguments should be provided
            dla_core (int): ID of target DLA core. If specified, no positional arguments should be provided.
            allow_gpu_fallback (bool): Allow TensorRT to schedule operations on GPU if they are not supported on DLA (ignored if device type is not DLA)

        Examples:
            - Device("gpu:1")
            - Device("cuda:1")
            - Device("dla:0", allow_gpu_fallback=True)
            - Device(gpu_id=0, dla_core=0, allow_gpu_fallback=True)
            - Device(dla_core=0, allow_gpu_fallback=True)
            - Device(gpu_id=1)
        """
        if len(args) == 1:
            if not isinstance(args[0], str):
                raise TypeError(
                    "When specifying Device through positional argument, argument must be str"
                )
            (self.device_type, id) = Device._parse_device_str(args[0])
            if self.device_type == trt.DeviceType.GPU:
                self.gpu_id = id
            else:
                self.dla_core = id
                self.gpu_id = 0
                logging.log(
                    logging.Level.Warning,
                    "Setting GPU id to 0 for device because device 0 manages DLA on Xavier",
                )

        elif not args:
            if "gpu_id" not in kwargs and "dla_core" not in kwargs:
                raise ValueError(
                    "Either gpu_id or dla_core or both must be defined if no string with device specs is provided as an arg"
                )

            if "dla_core" in kwargs:
                self.device_type = trt.DeviceType.DLA
                self.dla_core = kwargs["dla_core"]
                if "gpu_id" in kwargs:
                    self.gpu_id = kwargs["gpu_id"]
                else:
                    self.gpu_id = 0
                    logging.log(
                        logging.Level.Warning,
                        "Setting GPU id to 0 for device because device 0 manages DLA on Xavier",
                    )
            else:
                self.gpu_id = kwargs["gpu_id"]
                self.device_type = trt.DeviceType.GPU
        else:
            raise ValueError(
                f"Unexpected number of positional arguments for class Device \n    Found {len(args)} arguments, expected either zero or a single positional arguments"
            )

        if "allow_gpu_fallback" in kwargs:
            if not isinstance(kwargs["allow_gpu_fallback"], bool):
                raise TypeError("allow_gpu_fallback must be a bool")
            self.allow_gpu_fallback = kwargs["allow_gpu_fallback"]

    def __str__(self) -> str:
        return (
            f"Device(type={self.device_type}, gpu_id={self.gpu_id})"
            if self.device_type == trt.DeviceType.GPU
            else f", dla_core={self.dla_core}, allow_gpu_fallback={self.allow_gpu_fallback}"
        )

    def _to_internal(self) -> _C.Device:
        internal_dev = _C.Device()
        if self.device_type == trt.DeviceType.GPU:
            internal_dev.device_type = _C.DeviceType.GPU
        elif self.device_type == trt.DeviceType.DLA:
            internal_dev.device_type = _C.DeviceType.DLA
        else:
            raise ValueError(
                "Invalid DeviceType detected while parsing the Device class"
            )

        internal_dev.gpu_id = self.gpu_id
        internal_dev.dla_core = self.dla_core
        internal_dev.allow_gpu_fallback = self.allow_gpu_fallback
        return internal_dev

    def _to_serialized_rt_device(self) -> str:
        internal_dev = self._to_internal()
        return internal_dev._to_serialized_rt_device()

    @classmethod
    def _from_torch_device(cls, torch_dev: torch.device):
        if torch_dev.type != "cuda":
            raise ValueError('Torch Device specs must have type "cuda"')
        gpu_id = torch_dev.index
        return cls(gpu_id=gpu_id)

    @classmethod
    def _current_device(cls):
        try:
            dev = _C._get_current_device()
        except RuntimeError:
            logging.log(logging.Level.Error, "Cannot get current device")
            return None
        return cls(gpu_id=dev.gpu_id)

    @staticmethod
    def _parse_device_str(s):
        s = s.lower()
        spec = s.split(":")
        if spec[0] in ["gpu", "cuda"]:
            return (trt.DeviceType.GPU, int(spec[1]))
        elif spec[0] == "dla":
            return (trt.DeviceType.DLA, int(spec[1]))
