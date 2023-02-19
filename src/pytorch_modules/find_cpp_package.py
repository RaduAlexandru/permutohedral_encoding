import gc
import importlib
import warnings

import torch

ALL_COMPUTE_CAPABILITIES = [20, 21, 30, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 89, 90]

def find_package():

    if not torch.cuda.is_available():
        raise EnvironmentError("Unknown compute capability. Ensure PyTorch with CUDA support is installed.")

    def _get_device_compute_capability(idx):
        major, minor = torch.cuda.get_device_capability(idx)
        return major * 10 + minor

    def _get_system_compute_capability():
        num_devices = torch.cuda.device_count()
        device_capability = [_get_device_compute_capability(i) for i in range(num_devices)]
        system_capability = min(device_capability)

        if not all(cc == system_capability for cc in device_capability):
            warnings.warn(
                f"System has multiple GPUs with different compute capabilities: {device_capability}. "
                f"Using compute capability {system_capability} for best compatibility. "
                f"This may result in suboptimal performance."
            )
        return system_capability

    # Determine the capability of the system as the minimum of all
    # devices, ensuring that we have no runtime errors.
    system_compute_capability = _get_system_compute_capability()

    # Try to import the highest compute capability version of tcnn that
    # we can find and is compatible with the system's compute capability.
    _C = None
    for cc in reversed(ALL_COMPUTE_CAPABILITIES):
        if cc > system_compute_capability:
            # incompatible
            continue

        try:
            _C = importlib.import_module(f"permutohedral_encoding_bindings._{cc}_C")
            if cc != system_compute_capability:
                warnings.warn(f"tinycudann was built for lower compute capability ({cc}) than the system's ({system_compute_capability}). Performance may be suboptimal.")
            break
        except ModuleNotFoundError:
            pass

    if _C is None:
        raise EnvironmentError(f"Could not find compatible tinycudann extension for compute capability {system_compute_capability}.")

    return _C




# if not torch.cuda.is_available():
# 	raise EnvironmentError("Unknown compute capability. Ensure PyTorch with CUDA support is installed.")

# def _get_device_compute_capability(idx):
# 	major, minor = torch.cuda.get_device_capability(idx)
# 	return major * 10 + minor

# def _get_system_compute_capability():
# 	num_devices = torch.cuda.device_count()
# 	device_capability = [_get_device_compute_capability(i) for i in range(num_devices)]
# 	system_capability = min(device_capability)

# 	if not all(cc == system_capability for cc in device_capability):
# 		warnings.warn(
# 			f"System has multiple GPUs with different compute capabilities: {device_capability}. "
# 			f"Using compute capability {system_capability} for best compatibility. "
# 			f"This may result in suboptimal performance."
# 		)
# 	return system_capability

# # Determine the capability of the system as the minimum of all
# # devices, ensuring that we have no runtime errors.
# system_compute_capability = _get_system_compute_capability()

# # Try to import the highest compute capability version of tcnn that
# # we can find and is compatible with the system's compute capability.
# _C = None
# for cc in reversed(ALL_COMPUTE_CAPABILITIES):
# 	if cc > system_compute_capability:
# 		# incompatible
# 		continue

# 	try:
# 		_C = importlib.import_module(f"permutohedral_encoding_bindings._{cc}_C")
# 		if cc != system_compute_capability:
# 			warnings.warn(f"tinycudann was built for lower compute capability ({cc}) than the system's ({system_compute_capability}). Performance may be suboptimal.")
# 		break
# 	except ModuleNotFoundError:
# 		pass

# if _C is None:
# 	raise EnvironmentError(f"Could not find compatible tinycudann extension for compute capability {system_compute_capability}.")
