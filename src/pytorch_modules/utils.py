import torch
import math

#from nerfies
def cosine_easing_window(num_freqs, alpha):
	"""Eases in each frequency one by one with a cosine.
	This is equivalent to taking a Tukey window and sliding it to the right
	along the frequency spectrum.
	Args:
	num_freqs: the number of frequencies.
	alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.
	Returns:
	A 1-d numpy array with num_sample elements containing the window.
	"""
	x = torch.clip(alpha - torch.arange(num_freqs, dtype=torch.float32), 0.0, 1.0)
	return 0.5 * (1 + torch.cos(math.pi * x + math.pi))


def map_range_val( input_val, input_start, input_end,  output_start,  output_end):
	# input_clamped=torch.clamp(input_val, input_start, input_end)
	input_clamped=max(input_start, min(input_end, input_val))
	# input_clamped=torch.clamp(input_val, input_start, input_end)
	return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)