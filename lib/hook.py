import copy
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from opt_einsum import contract
from torch import nn, vmap
from torch.nn import Linear, EmbeddingBag, Conv2d, Conv1d, Conv3d, Embedding, GRU, LSTM, MultiheadAttention, \
	GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d

from lib.utils import cuda_device_name


# https://github.com/pytorch/opacus/blob/6d643ce93dedf47a76f768974277e13831699ec9/opacus/utils/tensor_utils.py#L1
def _filter_dilated_rows(
	tensor: torch.Tensor,
	dilation: Tuple[int, int, int],
	dilated_kernel_size: Tuple[int, int, int],
	kernel_size: Tuple[int, int, int],
):
	"""
	A helper function that removes extra rows created during the process of
	implementing dilation.

	Args:
			tensor: A tensor containing the output slices resulting from unfolding
							the input tensor to `unfold3d()`.
							Shape is ``(B, C, D_out, H_out, W_out, dilated_kernel_size[0],
							dilated_kernel_size[1], dilated_kernel_size[2])``.
			dilation: The dilation given to `unfold3d()`.
			dilated_kernel_size: The size of the dilated kernel.
			kernel_size: The size of the kernel given to `unfold3d()`.

	Returns:
			A tensor of shape (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
			For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`.

	Example:
			>>> tensor = torch.zeros([1, 1, 3, 3, 3, 5, 5, 5])
			>>> dilation = (2, 2, 2)
			>>> dilated_kernel_size = (5, 5, 5)
			>>> kernel_size = (3, 3, 3)
			>>> _filter_dilated_rows(tensor, dilation, dilated_kernel_size, kernel_size).shape
			torch.Size([1, 1, 3, 3, 3, 3, 3, 3])
	"""

	kernel_rank = len(kernel_size)

	indices_to_keep = [
		list(range(0, dilated_kernel_size[i], dilation[i])) for i in range(kernel_rank)
	]

	tensor_np = tensor.numpy()

	axis_offset = len(tensor.shape) - kernel_rank

	for dim in range(kernel_rank):
		tensor_np = np.take(tensor_np, indices_to_keep[dim], axis=axis_offset + dim)

	return torch.Tensor(tensor_np)

def _unfold2d(
	input,
	*,
	kernel_size: Tuple[int, int],
	padding: Union[str, Tuple[int, int]],
	stride: Tuple[int, int],
	dilation: Tuple[int, int],
):
	"""
	See :meth:`~torch.nn.functional.unfold`
	"""
	*shape, H, W = input.shape
	if padding == "same":
		total_pad_H = dilation[0] * (kernel_size[0] - 1)
		total_pad_W = dilation[1] * (kernel_size[1] - 1)
		pad_H_left = math.floor(total_pad_H / 2)
		pad_H_right = total_pad_H - pad_H_left
		pad_W_left = math.floor(total_pad_W / 2)
		pad_W_right = total_pad_W - pad_W_left

	elif padding == "valid":
		pad_W_left, pad_W_right, pad_H_left, pad_H_right = (0, 0, 0, 0)
	else:
		pad_H_left, pad_H_right, pad_W_left, pad_W_right = (
			padding[0],
			padding[0],
			padding[1],
			padding[1],
		)

	H_effective = (
									H
									+ pad_H_left
									+ pad_H_right
									- (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1))
								) // stride[0] + 1
	W_effective = (
									W
									+ pad_W_left
									+ pad_W_right
									+ -(kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1))
								) // stride[1] + 1
	# F.pad's first argument is the padding of the *last* dimension
	input = F.pad(input, (pad_W_left, pad_W_right, pad_H_left, pad_H_right))
	*shape_pad, H_pad, W_pad = input.shape
	strides = list(input.stride())
	strides = strides[:-2] + [
		W_pad * dilation[0],
		dilation[1],
		W_pad * stride[0],
		stride[1],
		]
	out = input.as_strided(
		shape + [kernel_size[0], kernel_size[1], H_effective, W_effective], strides
	)

	return out.reshape(input.size(0), -1, H_effective * W_effective)


def _unfold3d(
	tensor: torch.Tensor,
	*,
	kernel_size: Union[int, Tuple[int, int, int]],
	padding: Union[int, Tuple[int, int, int]] = 0,
	stride: Union[int, Tuple[int, int, int]] = 1,
	dilation: Union[int, Tuple[int, int, int]] = 1,
):
	r"""
	Extracts sliding local blocks from an batched input tensor.

	:class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
	This method implements the same action for 5D inputs

	Args:
			tensor: An input tensor of shape ``(B, C, D, H, W)``.
			kernel_size: the size of the sliding blocks
			padding: implicit zero padding to be added on both sides of input
			stride: the stride of the sliding blocks in the input spatial dimensions
			dilation: the spacing between the kernel points.

	Returns:
			A tensor of shape ``(B, C * np.product(kernel_size), L)``, where L - output spatial dimensions.
			See :class:`torch.nn.Unfold` for more details

	Example:
			>>> B, C, D, H, W = 3, 4, 5, 6, 7
			>>> tensor = torch.arange(1, B*C*D*H*W + 1.).view(B, C, D, H, W)
			>>> _unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
			torch.Size([3, 32, 120])
	"""

	if len(tensor.shape) != 5:
		raise ValueError(
			f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
		)

	if isinstance(kernel_size, int):
		kernel_size = (kernel_size, kernel_size, kernel_size)

	if isinstance(padding, int):
		padding = (padding, padding, padding)

	if isinstance(stride, int):
		stride = (stride, stride, stride)

	if isinstance(dilation, int):
		dilation = (dilation, dilation, dilation)

	if padding == "same":
		total_pad_D = dilation[0] * (kernel_size[0] - 1)
		total_pad_H = dilation[1] * (kernel_size[1] - 1)
		total_pad_W = dilation[2] * (kernel_size[2] - 1)
		pad_D_left = math.floor(total_pad_D / 2)
		pad_D_right = total_pad_D - pad_D_left
		pad_H_left = math.floor(total_pad_H / 2)
		pad_H_right = total_pad_H - pad_H_left
		pad_W_left = math.floor(total_pad_W / 2)
		pad_W_right = total_pad_W - pad_W_left

	elif padding == "valid":
		pad_D_left, pad_D_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right = (
			0,
			0,
			0,
			0,
			0,
			0,
		)
	else:
		pad_D_left, pad_D_right, pad_H_left, pad_H_right, pad_W_left, pad_W_right = (
			padding[0],
			padding[0],
			padding[1],
			padding[1],
			padding[2],
			padding[2],
		)

	batch_size, channels, _, _, _ = tensor.shape

	# Input shape: (B, C, D, H, W)
	tensor = F.pad(
		tensor,
		(pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right),
	)
	# Output shape: (B, C, D+pad_W_left+pad_W_right, H+pad_H_left+pad_H_right, W+pad_D_left+pad_D_right)

	dilated_kernel_size = (
		kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1),
		kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1),
		kernel_size[2] + (kernel_size[2] - 1) * (dilation[2] - 1),
	)

	tensor = tensor.unfold(dimension=2, size=dilated_kernel_size[0], step=stride[0])
	tensor = tensor.unfold(dimension=3, size=dilated_kernel_size[1], step=stride[1])
	tensor = tensor.unfold(dimension=4, size=dilated_kernel_size[2], step=stride[2])

	if dilation != (1, 1, 1):
		tensor = _filter_dilated_rows(tensor, dilation, dilated_kernel_size, kernel_size)

	# Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
	# For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`

	tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
	# Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

	tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(
		1, 2
	)
	# Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]

	return tensor


def has_trainable_params(module: nn.Module) -> bool:
	return any(p.requires_grad for p in module.parameters(recurse=False))


def make_per_sample_grad_layer_pure(layer):
	use_functorch = False
	# https://github.com/pytorch/opacus/blob/64680d16d95b1ec007f56b252c631ff786a4a9a7/opacus/grad_sample/linear.py#L26
	# consider https://medium.com/@mihirkhandekar/forward-and-backpropagation-in-grus-derived-deep-learning-5764f374f3f5
	if not use_functorch and not isinstance(layer, nn.GRU) and not isinstance(layer, LSTM) and not isinstance(layer, MultiheadAttention):
		if isinstance(layer, InstanceNorm1d) or isinstance(layer, InstanceNorm2d) or isinstance(layer, InstanceNorm3d):
			def per_sample_grad_fn(activations, backprops):
				activations = activations[0]
				ret = {}
				if layer.weight.requires_grad:
					gs = F.instance_norm(activations, eps=layer.eps) * backprops
					ret[layer.weight] = contract("ni...->ni", gs)
				if layer.bias is not None and layer.bias.requires_grad:
					ret[layer.bias] = contract("ni...->ni", backprops)
				return ret
		elif isinstance(layer, GroupNorm):
			def per_sample_grad_fn(activations, backprops):
				activations = activations[0]
				ret = {}
				if layer.weight.requires_grad:
					gs = F.group_norm(activations, layer.num_groups, eps=layer.eps) * backprops
					ret[layer.weight] = contract("ni...->ni", gs)
				if layer.bias is not None and layer.bias.requires_grad:
					ret[layer.bias] = contract("ni...->ni", backprops)
				return ret
		# elif isinstance(layer, LayerNorm):
		# 	def per_sample_grad_fn(activations, backprops):
		# 		activations = activations[0]
		# 		ret = {}
		# 		if layer.weight.requires_grad:
		# 			ret[layer.weight] = sum_over_all_but_batch_and_last_n(
		# 				F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
		# 				* backprops,
		# 				layer.weight.dim(),
		# 				)
		# 		if layer.bias.requires_grad:
		# 			ret[layer.bias] = sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim())
		# 		return ret
		elif isinstance(layer, Linear):
			def per_sample_grad_fn(activations, backprops):
				activations = activations[0]
				ret = {}
				if layer.weight.requires_grad:
					gs = contract("n...i,n...j->nij", backprops, activations)
					ret[layer.weight] = gs
				if layer.bias is not None and layer.bias.requires_grad:
					ret[layer.bias] = contract("n...k->nk", backprops)
				return ret
		elif isinstance(layer, EmbeddingBag):
			def per_sample_grad_fn(activations, backprops):
				index, offset = activations
				batch_size = offset.shape[0]
				gsm = torch.zeros(batch_size, layer.num_embeddings, layer.embedding_dim, device=cuda_device_name)

				for i in range(batch_size):
					begin = offset[i]
					if i < batch_size - 1:
						end = offset[i + 1]
					else:
						end = index.shape[0]

					if layer.mode == "sum":
						gsm[i][index[begin:end]] += backprops[i]
					elif layer.mode == "mean":
						gsm[i][index[begin:end]] += backprops[i] / (end - begin)

				ret = {}
				ret[layer.weight] = gsm

				return ret
		elif isinstance(layer, Conv2d) or isinstance(layer, Conv1d) or isinstance(layer, Conv3d):
			def per_sample_grad_fn(activations, backprops):
				activations = activations[0]
				n = activations.shape[0]
				if n == 0:
					# Empty batch
					ret = {}
					ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
					if layer.bias is not None and layer.bias.requires_grad:
						ret[layer.bias] = torch.zeros_like(layer.bias).unsqueeze(0)
					return ret

				# get activations and backprops in shape depending on the Conv layer
				if type(layer) == nn.Conv2d:
					activations = _unfold2d(
						activations,
						kernel_size=layer.kernel_size,
						padding=layer.padding,
						stride=layer.stride,
						dilation=layer.dilation,
					)
				elif type(layer) == nn.Conv1d:
					activations = activations.unsqueeze(-2)  # add the H dimension
					# set arguments to tuples with appropriate second element
					if layer.padding == "same":
						total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
						left_pad = math.floor(total_pad / 2)
						right_pad = total_pad - left_pad
					elif layer.padding == "valid":
						left_pad, right_pad = 0, 0
					else:
						left_pad, right_pad = layer.padding[0], layer.padding[0]
					activations = F.pad(activations, (left_pad, right_pad))
					activations = torch.nn.functional.unfold(
						activations,
						kernel_size=(1, layer.kernel_size[0]),
						stride=(1, layer.stride[0]),
						dilation=(1, layer.dilation[0]),
					)
				elif type(layer) == nn.Conv3d:
					activations = _unfold3d(
						activations,
						kernel_size=layer.kernel_size,
						padding=layer.padding,
						stride=layer.stride,
						dilation=layer.dilation,
					)
				backprops = backprops.reshape(n, -1, activations.shape[-1])

				ret = {}
				if layer.weight.requires_grad:
					# n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
					grad_sample = contract("noq,npq->nop", backprops, activations)
					# rearrange the above tensor and extract diagonals.
					grad_sample = grad_sample.view(
						n,
						layer.groups,
						-1,
						layer.groups,
						int(layer.in_channels / layer.groups),
						np.prod(layer.kernel_size),
					)
					# grad_sample = contract("ngrg...->ngr...", grad_sample).contiguous()
					grad_sample = contract("ngrg...->ngr...", grad_sample)
					shape = [n] + list(layer.weight.shape)
					ret[layer.weight] = grad_sample.view(shape)

				if layer.bias is not None and layer.bias.requires_grad:
					ret[layer.bias] = torch.sum(backprops, dim=2)

				return ret
		elif isinstance(layer, Embedding):
			def per_sample_grad_fn(activations, backprops):
				activations = activations[0]
				ret = {}
				if layer.weight.requires_grad:
					saved = torch.backends.cudnn.deterministic
					torch.backends.cudnn.deterministic = True

					batch_size = activations.shape[0]
					if batch_size == 0:
						ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
						return ret

					index = (
						activations.unsqueeze(-1)
						.expand(*activations.shape, layer.embedding_dim)
						.reshape(batch_size, -1, layer.embedding_dim)
					)
					grad_sample = torch.zeros(
						batch_size, *layer.weight.shape, device=layer.weight.device
					)
					grad_sample.scatter_add_(
						1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
					)
					torch.backends.cudnn.deterministic = saved
					ret[layer.weight] = grad_sample
				return ret
		else:
			raise NotImplementedError("Backprop for {} not implemented yet".format(type(layer)))
	else:
		stateless_layer = copy.deepcopy(layer)
		stateless_layer.to('meta')

		original_params = dict(layer.named_parameters())
		params = {k: v.detach() for k, v in layer.named_parameters()}
		buffers = {k: v.detach() for k, v in layer.named_buffers()}

		if isinstance(layer, GRU) or isinstance(layer, LSTM):
			def stateless_loss(a_params, a_buffers, inp, hidden, backprop):
				return (torch.func.functional_call(
					stateless_layer,
					(a_params, a_buffers),
					(inp.unsqueeze(0), hidden),
				)[0] * backprop).sum()

			# https://github.com/pytorch/pytorch/issues/96655
			mapped_grad_fn = vmap(
				torch.func.grad(stateless_loss),
				in_dims=(None, None, 0, None, 0),
				randomness='different',
			)

			def per_sample_grad_fn(sample_inputs, backprops):
				mapped_grad = mapped_grad_fn(
					params,
					buffers,
					sample_inputs[0],
					sample_inputs[1],
					backprops,
				)
				return {
					original_params[name]: per_sample_grad for name, per_sample_grad in mapped_grad.items()
				}
		elif isinstance(layer, MultiheadAttention):
			def stateless_loss(a_params, a_buffers, query, key, value, backprop):
				return (torch.func.functional_call(
					stateless_layer,
					(a_params, a_buffers),
					(query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)),  # multihead attention batch dim is 1
				)[0] * backprop).sum()

			# https://github.com/pytorch/pytorch/issues/96655
			mapped_grad_fn = vmap(
				torch.func.grad(stateless_loss),
				in_dims=(None, None, 1, 1, 1, 1),  # multihead attention batch dim is 1
				randomness='different',
			)

			def per_sample_grad_fn(sample_inputs, backprops):
				# print('\nshape', [inp.shape for inp in sample_inputs], backprops.shape)
				mapped_grad = mapped_grad_fn(
					params,
					buffers,
					sample_inputs[0],
					sample_inputs[1],
					sample_inputs[2],
					backprops,
				)
				return {
					original_params[name]: per_sample_grad for name, per_sample_grad in mapped_grad.items()
				}
		else:
			def stateless_loss(a_params, a_buffers, sample, backprop):
				return (torch.func.functional_call(
					stateless_layer,
					(a_params, a_buffers),
					sample.unsqueeze(0)
				)[0] * backprop).sum()

			# https://github.com/pytorch/pytorch/issues/96655
			mapped_grad_fn = vmap(
				torch.func.grad(stateless_loss),
				in_dims=(None, None, 0, 0),
				randomness='different',
			)

			def per_sample_grad_fn(sample_inputs, backprops):
				mapped_grad = mapped_grad_fn(
					params,
					buffers,
					sample_inputs[0],
					backprops,
				)
				return {
					original_params[name]: per_sample_grad for name, per_sample_grad in mapped_grad.items()
				}

	return per_sample_grad_fn
