

# print("modules in path", __file__)

# import gc
# import importlib
# import warnings

import torch

class PermutoEncodingFunc(Function):
    @staticmethod
    def forward(ctx, values, scale_per_level, positions, random_shift_per_level, anneal_window, concat_points, points_scaling, require_values_grad, require_positions_grad):

       
        sliced_values, splatting_indices, splatting_weights=Encoding.slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic(lattice_values_monolithic, scale_factor, positions, random_shift_per_level, anneal_window, concat_points, points_scaling, False, False )

        ctx.require_values_grad=require_values_grad
        ctx.require_positions_grad=require_positions_grad
        ctx.concat_points=concat_points
        ctx.lattice_structure = lattice_structure
        ctx.save_for_backward(lattice_values, positions, sliced_values, splatting_indices, splatting_weights, random_shift_per_level, anneal_window, scale_per_level)


        return sliced_values, splatting_indices, splatting_weights



       
    @staticmethod
    def backward(ctx, grad_sliced_values_monolithic, grad_splatting_indices, grad_splatting_weights):
        

        require_lattice_values_grad=ctx.require_lattice_values_grad
        require_positions_grad=ctx.require_positions_grad
        assert require_lattice_values_grad or require_positions_grad, "We cannot perform the backward function on the slicing because we did not precompute the required tensors in the forward pass. To enable this, set the model.train(), set torch.set_grad_enabled(True) and make lattice_values have required_grad=True"
      
        lattice_values_monolithic, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors


        # if require_lattice_values_grad and not require_positions_grad:
        #     # lattice_values_monolithic, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors
        #     # print("getting from backward without LV")
        #     positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors
        #     lattice_values_monolithic=None
        # elif require_lattice_values_grad and require_positions_grad:
        #     # positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors
        #     # print("getting from backward with LV. ctx has len", len(ctx.saved_tensors))
        #     lattice_values_monolithic, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors


        lattice_structure = ctx.lattice_structure
        # sigma=ctx.sigma
        # sigmas_list=ctx.sigmas_list
        concat_points=ctx.concat_points
        # lattice_structure.set_sigma(sigma)


        return SliceLatticeWithCollisionFastMRMonolithicBackward.apply(grad_sliced_values_monolithic, lattice_values_monolithic, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, lattice_structure, scale_factor, concat_points,  require_lattice_values_grad, require_positions_grad) 


        # lattice_values=None
        # if require_lattice_values_grad:
        #     grad_sliced_values_monolithic=grad_sliced_values_monolithic.contiguous()
        #     lattice_values=lattice_structure.slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic(positions, grad_sliced_values_monolithic, splatting_indices, splatting_weights) 
        #     lattice_values=lattice_values.permute(0,2,1)
       
        # grad_positions=None
        # if require_positions_grad:
        #     grad_positions=torch.zeros_like(positions)


        # # ctx.lattice_structure=0 # release the pointer to this so it gets cleaned up

        # return lattice_values, None, None, grad_positions, None, None, None, None


#in order to enable a double backward like in https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class SliceLatticeWithCollisionFastMRMonolithicBackward(Function):
    @staticmethod
    def forward(ctx, grad_sliced_values_monolithic, lattice_values_monolithic, positions, sliced_values_hom, splatting_indices, splatting_weights, random_shift_monolithic, anneal_window, lattice_structure, scale_factor, concat_points,  require_lattice_values_grad, require_positions_grad):
        lattice_values=None
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

            if require_positions_grad:
                TIME_START("slice_back_posgrad")


            # print("grad_sliced_values_monolithic min max ",grad_sliced_values_monolithic.min(), grad_sliced_values_monolithic.max())
            #debug
            # if require_positions_grad:
                # require_lattice_values_grad=False



            ctx.save_for_backward(lattice_values_monolithic, grad_sliced_values_monolithic, positions, random_shift_monolithic, anneal_window, scale_factor )
            ctx.concat_points=concat_points
            ctx.lattice_structure = lattice_structure

            # print("require_positions_grad",require_positions_grad)

            lattice_values_grad, positions_grad=lattice_structure.slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(positions, lattice_values_monolithic, grad_sliced_values_monolithic, scale_factor, random_shift_monolithic, anneal_window, concat_points, require_lattice_values_grad, require_positions_grad) 
            if Lattice.is_half_precision():
                pass
            else:
                # lattice_values=lattice_values.permute(0,2,1)
                pass
            if require_positions_grad:
                TIME_END("slice_back_posgrad")

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
        
        return lattice_values_grad, None, None, positions_grad, None, None, None, None, None, None, None, None, None
    @staticmethod
    def backward(ctx, double_lattice_values_grad, dumm2, dummy3, double_positions_grad, dummy5, dummy6, dummy7, dummy8, dumm9, dummy10, dummy11, dummy12, dummy13):

        #in the forward pass of this module we do 
        #lattice_values_grad, positions_grad = slice_back(lattice_values_monolithic, grad_sliced_values_monolithic, positions)
        #now in the backward pass we have the upstream gradient which is double_lattice_values_grad, double_positions_grad
        #we want to propagate the double_positions_grad into lattice_values_monolithic and grad_sliced_values_monolithic

        lattice_values_monolithic, grad_sliced_values_monolithic, positions, random_shift_monolithic, anneal_window, scale_factor =ctx.saved_tensors
        concat_points=ctx.concat_points
        lattice_structure = ctx.lattice_structure

        # print("concat_points",concat_points)

        grad_lattice_values_monolithic, grad_grad_sliced_values_monolithic=lattice_structure.slice_double_back_from_positions_grad(double_positions_grad, positions, lattice_values_monolithic, grad_sliced_values_monolithic, scale_factor, random_shift_monolithic, anneal_window, concat_points )

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
