import torch
from permutohedral_encoding.pytorch_modules.find_cpp_package import *
from permutohedral_encoding.pytorch_modules.funcs import *
from permutohedral_encoding.pytorch_modules.utils import cosine_easing_window
import math

_C=find_package()


class PermutoEncoding(torch.nn.Module):
	def __init__(self, pos_dim, capacity, nr_levels, nr_feat_per_level, scale_per_level, appply_random_shift_per_level=True, concat_points=False, concat_points_scaling=1.0):
		super(PermutoEncoding, self).__init__()
		self.pos_dim=pos_dim 
		self.capacity=capacity 
		self.nr_levels=nr_levels 
		self.nr_feat_per_level=nr_feat_per_level
		self.scale_per_level=scale_per_level
		self.appply_random_shift_per_level=appply_random_shift_per_level
		self.concat_points=concat_points 
		self.concat_points_scaling=concat_points_scaling

		

		#create hashmap values
		lattice_values=torch.randn( capacity, nr_levels, nr_feat_per_level )*1e-5
		lattice_values=lattice_values.permute(1,0,2).contiguous() #makes it nr_levels x capacity x nr_feat
		self.lattice_values=torch.nn.Parameter(lattice_values.cuda())

		#each levels of the hashamp can be randomly shifted so that we minimize collisions
		if appply_random_shift_per_level:
			random_shift_per_level=torch.randn( nr_levels, pos_dim)*10
			self.random_shift_per_level=torch.nn.Parameter( random_shift_per_level.cuda() ) #we make it a parameter just so it gets saved when we checkpoint
		else:
			self.random_shift_per_level= torch.nn.Parameter( torch.empty((1)).cuda() )


		#make a anneal window of all ones 
		self.anneal_window=torch.ones((nr_levels)).cuda()


		#per_lvl multiplier. Makes the convergence slightly faster as it can easily dampen frequencies that are not required for the current scene
		#howver also makes the slicing almost twice as slow
		# self.per_lvl_multiplier=torch.nn.Parameter( torch.ones((nr_levels)).cuda() )


		#make the lattice wrapper
		self.fixed_params=_C.EncodingFixedParams(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_per_level, self.random_shift_per_level, self.concat_points, self.concat_points_scaling)
		self.lattice=_C.EncodingWrapper.create(self.pos_dim, self.nr_feat_per_level, self.fixed_params)


		#the first time we run the module we set some constant on the gpu. this a somewhat costly oepration because it requires a sync to cpu. so we do it only once
		self.lattice.copy_to_constant_mem(self.fixed_params)
		
	def forward(self, positions, anneal_window=None):

		nr_positions=positions.shape[0]

		# pad to multiple of 128 bytes in order to enable coalesced accesses
		# nr_bytes_per_col=nr_positions*4
		# def round_to_multiple(number, multiple):
		# 	return multiple * round(number / multiple)
		# nr_bytes_per_col=round_to_multiple(nr_bytes_per_col,128)
		# nr_positions_padded=int(nr_bytes_per_col/4)
		# #pad
		# if nr_positions!=nr_positions_padded:
		# 	positions_pad=torch.nn.functional.pad(positions, [0, 0, 0, nr_positions_padded - nr_positions])	



		# positions=positions.transpose(0,1).contiguous().transpose(0,1)






		# check for posdim
		assert positions.shape[1] == self.pos_dim,"Pos dim for the lattice doesn't correspond with the position of the points."

		if anneal_window is None:
			anneal_window=self.anneal_window
		else:
			anneal_window=anneal_window.cuda()
		

		require_lattice_values_grad= self.lattice_values.requires_grad and torch.is_grad_enabled()
		require_positions_grad=  positions.requires_grad and torch.is_grad_enabled()


		# print("self.per_lvl_multiplier", self.per_lvl_multiplier.min(), self.per_lvl_multiplier.max())
		# print("per_lvl_multiplier",self.per_lvl_multiplier)

		sliced_values= PermutoEncodingFunc.apply(self.lattice, self.lattice_values, positions, anneal_window, require_lattice_values_grad, require_positions_grad)

		sliced_values=sliced_values.permute(2,0,1).reshape(nr_positions, -1) #from lvl, val, nr_positions to nr_positions x lvl x val
		# sliced_values=sliced_values.view(nr_positions, -1)
		# sliced_values=sliced_values.view(-1,nr_positions).transpose(0,1)



		return sliced_values

	def output_dims(self):
		#if we concat also the points, we add a series of extra resolutions to contain those points
		nr_resolutions_extra=0;
		if self.concat_points:
			nr_resolutions_extra=math.ceil(float(self.pos_dim)/self.nr_feat_per_level)

		out_dims=self.nr_feat_per_level*(self.nr_levels + nr_resolutions_extra)

		return out_dims




#coarse2fine  which slowly anneals the weights of a vector of size nr_values. t is between 0 and 1
class Coarse2Fine(torch.nn.Module):
	def __init__(self, nr_values):  
		super(Coarse2Fine, self).__init__()
		
		self.nr_values=nr_values 
		self.last_t=0

	def forward(self, t):

		alpha=t*self.nr_values #becasue cosine_easing_window except the alpha to be in range 0, nr_values
		window=cosine_easing_window(self.nr_values, alpha)

		self.last_t=t
		assert t<=1.0,"t cannot be larger than 1.0"

		return window

	def get_last_t(self):
		return self.last_t


