import torch
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
		# self.scale_factor=_C.Encoding.compute_scale_factor_tensor(scale_per_level,pos_dim).cuda() 
		# self.scale_factor=torch.tensor((1)).cuda() 

		#create hashmap values
		lattice_values=torch.randn( capacity, nr_levels, nr_feat_per_level )*1e-5
		lattice_values=lattice_values.permute(1,0,2).contiguous() #makes it nr_levels x capacity x nr_feat
		self.lattice_values=torch.nn.Parameter(lattice_values.cuda())
		# self.register_parameter(name="lattice_values", param=self.lattice_values)

		#each levels of the hashamp can be randomly shifted so that we minimize collisions
		if appply_random_shift_per_level:
			random_shift_per_level=torch.randn( nr_levels, 3)*10
			self.random_shift_per_level=torch.nn.Parameter( random_shift_per_level.cuda() ) #we make it a parameter just so it gets saved when we checkpoint
		else:
			self.random_shift_per_level= torch.nn.Parameter( torch.empty((1)).cuda() )
		# self.register_parameter(name="random_shift_per_level", param=self.random_shift_per_level)


		#make a anneal window of all ones 
		self.anneal_window=torch.ones((nr_levels)).cuda()


		#make the lattice wrapper
		# print("making lattice")
		fixed_params=_C.EncodingFixedParams(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_per_level, self.random_shift_per_level, self.concat_points, self.concat_points_scaling)
		self.lattice=_C.EncodingWrapper.create(self.pos_dim, self.nr_feat_per_level, fixed_params)
		# print("made lattice")
		
	def forward(self, positions, anneal_window=None):

		nr_positions=positions.shape[0]

		# check for posdim
		assert positions.shape[1] == self.pos_dim,"Pos dim for the lattice doesn't correspond with the position of the points."

		if anneal_window is None:
			anneal_window=self.anneal_window
		else:
			anneal_window=anneal_window.cuda()
		

		require_lattice_values_grad= self.lattice_values.requires_grad and torch.is_grad_enabled()
		require_positions_grad=  positions.requires_grad and torch.is_grad_enabled()

		
		# input_struct=_C.EncodingInput(self.lattice_values, self.positions, anneal_window, require_lattice_values_grad, require_positions_grad)
		# sliced_values, splatting_indices, splatting_weights= PermutoEncodingFunc.apply(self.lattice_values, self.scale_factor, positions, self.random_shift_per_level, anneal_window, self.concat_points, self.concat_points_scaling, require_lattice_values_grad, require_positions_grad)
		sliced_values, splatting_indices, splatting_weights= PermutoEncodingFunc.apply(self.lattice, self.lattice_values, positions, anneal_window, require_lattice_values_grad, require_positions_grad)

		sliced_values=sliced_values.permute(2,0,1).reshape(nr_positions, -1) #from lvl, val, nr_positions to nr_positions x lvl x val

		# return sliced_values, splatting_indices, splatting_weights 
		return sliced_values


