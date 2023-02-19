

# print("modules in path", __file__)

import gc
import importlib
import warnings

import torch

# ALL_COMPUTE_CAPABILITIES = [20, 21, 30, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 89, 90]

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

# def _torch_precision(tcnn_precision):
# 	if tcnn_precision == _C.Precision.Fp16:
# 		return torch.half
# 	elif tcnn_precision == _C.Precision.Fp32:
# 		return torch.float
# 	else:
# 		raise ValueError(f"Unknown precision {tcnn_precision}")

# def free_temporary_memory():
# 	# Ensure all Python objects (potentially pointing
# 	# to temporary TCNN allocations) are cleaned up.
# 	gc.collect()
# 	_C.free_temporary_memory()

# def null_tensor_like(tensor):
# 	return torch.empty([], dtype=tensor.dtype, device=tensor.device)

# def null_tensor_to_none(tensor):
# 	if len(tensor.shape) == 0:
# 		return None
# 	return tensor


# class Encoding(torch.nn.Module):
#     def __init__(self):
#         print("making")



from permutohedral_encoding.pytorch_modules.find_cpp_package import *
from permutohedral_encoding.pytorch_modules.funcs import *

_C=find_package()

class PermutoEncoding(torch.nn.Module):
	def __init__(self, pos_dim, capacity, nr_levels, nr_feat_per_level, scale_per_level, appply_random_shift_per_level, concat_points, concat_points_scaling):
		super(PermutoEncoding, self).__init__()
		self.pos_dim=pos_dim 
		self.capacity=capacity 
		self.nr_levels=nr_levels 
		self.nr_feat_per_level=nr_feat_per_level
		self.scale_per_level=scale_per_level
		self.appply_random_shift_per_level=appply_random_shift_per_level
		self.concat_points=concat_points 
		self.concat_points_scaling=concat_points_scaling

		#create the scale factor
		self.scale_factor=_C.Encoding.compute_scale_factor_tensor(scale_per_level,pos_dim).cuda() 

		#create hashmap values
		self.lattice_values=torch.randn( capacity, nr_levels, nr_feat_per_level )*1e-5
		self.lattice_values=self.lattice_values.permute(1,0,2).contiguous() #makes it nr_levels x capacity x nr_feat
		self.lattice_values=torch.nn.Parameter(self.lattice_values).cuda() 

		#each levels of the hashamp can be randomly shifted so that we minimize collisions
		self.random_shift_per_level=torch.empty((1))
		if appply_random_shift_per_level:
			self.random_shift_per_level=torch.randn( nr_levels, 3)*10
			self.random_shift_per_level=torch.nn.Parameter( self.random_shift_per_level ).cuda() #we make it a parameter just so it gets saved when we checkpoint


		#make a anneal window of all ones 
		self.anneal_window=torch.ones((nr_levels)).cuda()
		
		
	def forward(self, positions, anneal_window=None):

		nr_positions=positions.shape[0]

		#TODO 
		# check for posdim
		assert positions.shape[1] == self.pos_dim,"Pos dim for the lattice doesn't correspond with the position of the points."

		if anneal_window is None:
			anneal_window=self.anneal_window
		

		require_lattice_values_grad= self.lattice_values.requires_grad and torch.is_grad_enabled()
		require_positions_grad=  positions.requires_grad and torch.is_grad_enabled()

		sliced_values, splatting_indices, splatting_weights= PermutoEncodingFunc.apply(self.lattice_values, self.scale_factor, positions, self.random_shift_per_level, anneal_window, self.concat_points, self.concat_points_scaling, require_lattice_values_grad, require_positions_grad)

		sliced_values=sliced_values.permute(2,0,1).reshape(nr_positions, -1) #from lvl, val, nr_positions to nr_positions x lvl x val

		# return sliced_values, splatting_indices, splatting_weights 
		return sliced_values


# class SliceLatticeWithCollisionsFastMRMonolithicModule(torch.nn.Module):
#     def __init__(self):
#         super(SliceLatticeWithCollisionsFastMRMonolithicModule, self).__init__()
#     def forward(self, lattice_values_monolithic, scale_factor, lattice_structure, positions, random_shift_monolithic, anneal_window, concat_points, points_scaling):

#         nr_positions=positions.shape[0]
#         nr_resolutions=lattice_values_monolithic.shape[0]
#         nr_lattice_vertices=lattice_values_monolithic.shape[1]
#         nr_lattice_features=lattice_values_monolithic.shape[2]

#         # require_lattice_values_grad= self.training and lattice_values_monolithic.requires_grad and torch.is_grad_enabled()
#         # require_positions_grad= self.training and positions.requires_grad and torch.is_grad_enabled()
#         require_lattice_values_grad= lattice_values_monolithic.requires_grad and torch.is_grad_enabled()
#         require_positions_grad=  positions.requires_grad and torch.is_grad_enabled()
#         # require_lattice_values_grad=True

#         sliced_values_monolithic, splatting_indices, splatting_weights= SliceLatticeWithCollisionFastMRMonolithic.apply(lattice_values_monolithic, scale_factor, lattice_structure, positions, random_shift_monolithic, anneal_window, concat_points, points_scaling, require_lattice_values_grad, require_positions_grad)

#         # print("sliced_values_monolithic",sliced_values_monolithic)
#         # print("sliced_values_monolithic shape",sliced_values_monolithic.shape)
#         # print("sliced_values_monolithic last 3 dimensions",sliced_values_monolithic[23:25, :,:])

#         sliced_values_monolithic=sliced_values_monolithic.permute(2,0,1).reshape(nr_positions, -1) #from lvl, val, nr_positions to nr_positions x lvl x val

#         return sliced_values_monolithic, splatting_indices, splatting_weights

