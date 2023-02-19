

# print("modules in path", __file__)

# import gc
# import importlib
# import warnings

import torch
# from permutohedral_encoding import Encoding
from permutohedral_encoding.pytorch_modules.find_cpp_package import *

_C=find_package()


class PermutoEncodingFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, lattice_values, scale_factor, positions, random_shift_per_level, anneal_window, concat_points, points_scaling, require_lattice_values_grad, require_positions_grad):

		# print( "values 0:", type(values))
		# print( "scale_factor 1:", type(scale_factor))
		# print( "positions 2:", type(positions))
		# print( "random_shift_per_level 3:", type(random_shift_per_level))
		# print( "anneal_window 4:", type(anneal_window))
		# print( "concat_points 5:", type(concat_points))
		# print( "points_scaling 6:", type(points_scaling))


		# #test
		# test_tensor=torch.ones((1))
		# _C.Encoding.test(test_tensor) 
		# print("finished test")


	   
		sliced_values, splatting_indices, splatting_weights=_C.Encoding.slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic(lattice_values, scale_factor, positions, random_shift_per_level, anneal_window, concat_points, points_scaling, False, False )

		ctx.require_lattice_values_grad=require_lattice_values_grad
		ctx.require_positions_grad=require_positions_grad
		ctx.concat_points=concat_points
		# ctx.lattice_structure = lattice_structure
		ctx.save_for_backward(lattice_values, positions, sliced_values, splatting_indices, splatting_weights, random_shift_per_level, anneal_window, scale_factor)


		return sliced_values, splatting_indices, splatting_weights



	   
	@staticmethod
	def backward(ctx, grad_sliced_values_monolithic, grad_splatting_indices, grad_splatting_weights):


		# return None, None, None, None, None, None, None, None, None, None, None, None, None


		require_lattice_values_grad=ctx.require_lattice_values_grad
		require_positions_grad=ctx.require_positions_grad
		assert require_lattice_values_grad or require_positions_grad, "We cannot perform the backward function on the slicing because we did not precompute the required tensors in the forward pass. To enable this, set the model.train(), set torch.set_grad_enabled(True) and make lattice_values have required_grad=True"
	  
		lattice_values, positions, sliced_values, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors


		# lattice_structure = ctx.lattice_structure
		# sigma=ctx.sigma
		# sigmas_list=ctx.sigmas_list
		concat_points=ctx.concat_points
		# lattice_structure.set_sigma(sigma)


		return SliceLatticeWithCollisionFastMRMonolithicBackward.apply(grad_sliced_values_monolithic, lattice_values, positions, sliced_values, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window,  scale_factor, concat_points,  require_lattice_values_grad, require_positions_grad) 


	   


#in order to enable a double backward like in https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class SliceLatticeWithCollisionFastMRMonolithicBackward(torch.autograd.Function):
	@staticmethod
	def forward(ctx, grad_sliced_values_monolithic, lattice_values, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor, concat_points,  require_lattice_values_grad, require_positions_grad):
		# lattice_values=None
		if require_lattice_values_grad or require_positions_grad:
			grad_sliced_values_monolithic=grad_sliced_values_monolithic.contiguous()

			# print("grad_sliced_values_monolithic", grad_sliced_values_monolithic.type())
			# print("grad_sliced_values_monolithic min max", grad_sliced_values_monolithic.min(), grad_sliced_values_monolithic.max())
			# print("grad_sliced_values_monolithic", grad_sliced_values_monolithic)
			# exot(1)
			
			#try some stuff
			# grad_sliced_values_monolithic=grad_sliced_values_monolithic.transpose(1,2).contiguous().transpose(1,2)
			# splatting_indices=splatting_indices.transpose(1,2).contiguous().transpose(1,2)
			# splatting_weights=splatting_weights.transpose(1,2).contiguous().transpose(1,2)

			# print("require_lattice_values_grad",require_lattice_values_grad)
			# print("require_positions_grad",require_positions_grad)

			# positions_grad=None
			# lattice_values_grad=lattice_structure.slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic(positions, grad_sliced_values_monolithic, splatting_indices, splatting_weights) 

			# if require_positions_grad:
				# TIME_START("slice_back_posgrad")


			# print("grad_sliced_values_monolithic min max ",grad_sliced_values_monolithic.min(), grad_sliced_values_monolithic.max())
			#debug
			# if require_positions_grad:
				# require_lattice_values_grad=False



			ctx.save_for_backward(lattice_values, grad_sliced_values_monolithic, positions, random_shift_monolithic, anneal_window, scale_factor )
			ctx.concat_points=concat_points
			# ctx.lattice_structure = lattice_structure

			# print("require_positions_grad",require_positions_grad)

			# print("running backward")
			lattice_values_grad, positions_grad=_C.Encoding.slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(positions, lattice_values, grad_sliced_values_monolithic, scale_factor, random_shift_monolithic, anneal_window, concat_points, require_lattice_values_grad, require_positions_grad) 
			# print("finished backward")
			# if Lattice.is_half_precision():
			# 	pass
			# else:
			# 	# lattice_values=lattice_values.permute(0,2,1)
			# 	pass
			# if require_positions_grad:
			# 	TIME_END("slice_back_posgrad")

			# if require_positions_grad:
				# require_lattice_values_grad=False
				# lattice_values_grad=None

			# print("aacumulated gradient is min max ", lattice_values.min(), lattice_values.max())
	   
		# grad_positions=None
		# if require_positions_grad:
			# grad_positions=torch.zeros_like(positions)
			# grad_positions=None

		# positions_grad=torch.zeros_like(positions)

		# print("require_positions_grad", require_positions_grad)
		# print("func positions_grad min max", positions_grad.min(), positions_grad.max())

		# print("grad_sliced_values_monolithic min max ", grad_sliced_values_monolithic.min(), grad_sliced_values_monolithic.max())
		# print("lattice values grad is ", lattice_values_grad.min(), lattice_values_grad.max())
		# print("positions_grad is ", positions_grad.min(), positions_grad.max())

		# if torch.isnan(grad_sliced_values_monolithic).any():
		#     print("wtf ")
		#     exit()
		# if torch.isnan(lattice_values_grad).any():
		#     print("wtf ")
		#     exit()
		# if torch.isnan(positions_grad).any():
		#     print("wtf ")
		#     exit()

		# print("positions_grad ", positions_grad.shape)
		# print("lattice_values_grad ", lattice_values_grad.shape)
		
		return lattice_values_grad, None, positions_grad, None, None, None, None, None, None, None, None, None
	@staticmethod
	def backward(ctx, double_lattice_values_grad, dumm2, dummy3, double_positions_grad, dummy5, dummy6, dummy7, dummy8, dumm9, dummy10, dummy11, dummy12, dummy13):

		# print("double back------")

		#in the forward pass of this module we do 
		#lattice_values_grad, positions_grad = slice_back(lattice_values_monolithic, grad_sliced_values_monolithic, positions)
		#now in the backward pass we have the upstream gradient which is double_lattice_values_grad, double_positions_grad
		#we want to propagate the double_positions_grad into lattice_values_monolithic and grad_sliced_values_monolithic

		lattice_values, grad_sliced_values_monolithic, positions, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors
		concat_points=ctx.concat_points
		# lattice_structure = ctx.lattice_structure

		# print("concat_points",concat_points)

		grad_lattice_values_monolithic, grad_grad_sliced_values_monolithic=_C.Encoding.slice_double_back_from_positions_grad(double_positions_grad, positions, values, grad_sliced_values_monolithic, scale_factor, random_shift_monolithic, anneal_window, concat_points )

		# grad_lattice_values_monolithic=grad_lattice_values_monolithic*0
		# grad_grad_sliced_values_monolithic=grad_grad_sliced_values_monolithic*0

		# print("double_lattice_values_grad min max ", double_lattice_values_grad.min(), double_lattice_values_grad.max())
		# print("double_positions_grad min max ", double_positions_grad.min(), double_positions_grad.max())
		# print("grad_lattice_values_monolithic is ", grad_lattice_values_monolithic.min(), grad_lattice_values_monolithic.max())
		# print("grad_grad_sliced_values_monolithic is ", grad_grad_sliced_values_monolithic.min(), grad_grad_sliced_values_monolithic.max())

		# if torch.isnan(double_lattice_values_grad).any():
		#     print("wtf ")
		#     exit() 
		# if torch.isnan(double_positions_grad).any():
		#     print("wtf ")
		#     exit() 
		# if torch.isnan(grad_lattice_values_monolithic).any():
		#     print("wtf ")
		#     exit() 
		# if torch.isnan(grad_grad_sliced_values_monolithic).any():
		#     print("wtf ")
		#     exit() 
		
		return grad_grad_sliced_values_monolithic, grad_lattice_values_monolithic, None, None, None, None, None, None, None, None, None, None, None
