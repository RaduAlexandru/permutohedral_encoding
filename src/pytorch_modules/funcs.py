import torch
from permutohedral_encoding.pytorch_modules.find_cpp_package import *

_C=find_package()



class PermutoEncodingFunc(torch.autograd.Function):
	@staticmethod
	# def forward(ctx, lattice_values, scale_factor, positions, random_shift_per_level, anneal_window, concat_points, points_scaling, require_lattice_values_grad, require_positions_grad):
	def forward(ctx, lattice, lattice_values, positions, anneal_window, require_lattice_values_grad, require_positions_grad):


	   
		input_struct=_C.EncodingInput(lattice_values, positions, anneal_window, require_lattice_values_grad, require_positions_grad)
		sliced_values, splatting_indices, splatting_weights=lattice.forward(input_struct )

		ctx.lattice=lattice
		ctx.input_struct=input_struct
		ctx.save_for_backward(sliced_values, splatting_indices, splatting_weights)


		return sliced_values, splatting_indices, splatting_weights



	   
	@staticmethod
	def backward(ctx, grad_sliced_values_monolithic, grad_splatting_indices, grad_splatting_weights):


		lattice=ctx.lattice
		input_struct=ctx.input_struct

		assert input_struct.m_require_lattice_values_grad or input_struct.m_require_positions_grad, "We cannot perform the backward function on the slicing because we did not precompute the required tensors in the forward pass. To enable this, set the model.train(), set torch.set_grad_enabled(True) and make lattice_values have required_grad=True"
	  
		sliced_values, splatting_indices, splatting_weights =ctx.saved_tensors


		#we pass the tensors of lattice_values and positiosn explicitly and not throught the input struct so that we can compute gradients from them for the double backward pass
		return SliceLatticeWithCollisionFastMRMonolithicBackward.apply(lattice, input_struct, grad_sliced_values_monolithic, input_struct.m_lattice_values, input_struct.m_positions_raw, sliced_values, splatting_indices, splatting_weights) 


	   

# in order to enable a double backward like in https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class SliceLatticeWithCollisionFastMRMonolithicBackward(torch.autograd.Function):
	@staticmethod
	def forward(ctx, lattice, input_struct, grad_sliced_values_monolithic, lattice_values, positions, sliced_values_hom, splatting_indices, splatting_weights):

		# print("in backward")

		lattice_values_grad=None
		positions_grad=None
		
		if input_struct.m_require_lattice_values_grad or input_struct.m_require_positions_grad:
			grad_sliced_values_monolithic=grad_sliced_values_monolithic.contiguous()

			ctx.save_for_backward(grad_sliced_values_monolithic)
			ctx.lattice=lattice
			ctx.input_struct=input_struct
			
			lattice_values_grad, positions_grad=lattice.backward(input_struct, grad_sliced_values_monolithic) 

			# print("grad_sliced_values_monolithic", grad_sliced_values_monolithic.min(), grad_sliced_values_monolithic.max())
			# print("lattice_values_grad min max", lattice_values_grad.min(), lattice_values_grad.max())	
			# print("positions_grad min max", positions_grad.min(), positions_grad.max())	
		
		return None, lattice_values_grad, positions_grad, None, None, None, None, None, None, None, None, None
	@staticmethod
	def backward(ctx, dummy1, double_lattice_values_grad, double_positions_grad, dummy5, dummy6, dummy7, dummy8, dumm9, dummy10, dummy11, dummy12, dummy13):


		#in the forward pass of this module we do 
		#lattice_values_grad, positions_grad = slice_back(lattice_values_monolithic, grad_sliced_values_monolithic, positions)
		#now in the backward pass we have the upstream gradient which is double_lattice_values_grad, double_positions_grad
		#we want to propagate the double_positions_grad into lattice_values_monolithic and grad_sliced_values_monolithic

		grad_sliced_values_monolithic, =ctx.saved_tensors
		lattice=ctx.lattice
		input_struct=ctx.input_struct


		grad_lattice_values_monolithic, grad_grad_sliced_values_monolithic=lattice.double_backward(input_struct, double_positions_grad,grad_sliced_values_monolithic )

		
		return None, None, grad_grad_sliced_values_monolithic, grad_lattice_values_monolithic, None, None, None, None, None, None, None, None, None, None
		# return None, None, None, None, None, None, None, None, None, None, None, None, None, None




#in order to enable a double backward like in https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
# class SliceLatticeWithCollisionFastMRMonolithicBackward(torch.autograd.Function):
# 	@staticmethod
# 	def forward(ctx, grad_sliced_values_monolithic, lattice_values, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor, concat_points,  require_lattice_values_grad, require_positions_grad):

# 		lattice_values_grad=None
# 		positions_grad=None
		
# 		if require_lattice_values_grad or require_positions_grad:
# 			grad_sliced_values_monolithic=grad_sliced_values_monolithic.contiguous()

# 			ctx.save_for_backward(lattice_values, grad_sliced_values_monolithic, positions, random_shift_monolithic, anneal_window, scale_factor )
# 			ctx.concat_points=concat_points
			
# 			lattice_values_grad, positions_grad=_C.Encoding.slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(positions, lattice_values, grad_sliced_values_monolithic, scale_factor, random_shift_monolithic, anneal_window, concat_points, require_lattice_values_grad, require_positions_grad) 
			
		
# 		return lattice_values_grad, None, positions_grad, None, None, None, None, None, None, None, None, None
# 	@staticmethod
# 	def backward(ctx, double_lattice_values_grad, dumm2, double_positions_grad, dummy5, dummy6, dummy7, dummy8, dumm9, dummy10, dummy11, dummy12, dummy13):


# 		#in the forward pass of this module we do 
# 		#lattice_values_grad, positions_grad = slice_back(lattice_values_monolithic, grad_sliced_values_monolithic, positions)
# 		#now in the backward pass we have the upstream gradient which is double_lattice_values_grad, double_positions_grad
# 		#we want to propagate the double_positions_grad into lattice_values_monolithic and grad_sliced_values_monolithic

# 		lattice_values, grad_sliced_values_monolithic, positions, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors
# 		concat_points=ctx.concat_points


# 		grad_lattice_values_monolithic, grad_grad_sliced_values_monolithic=_C.Encoding.slice_double_back_from_positions_grad(double_positions_grad, positions, lattice_values, grad_sliced_values_monolithic, scale_factor, random_shift_monolithic, anneal_window, concat_points )

		
# 		return grad_grad_sliced_values_monolithic, grad_lattice_values_monolithic, None, None, None, None, None, None, None, None, None, None
