from math import prod

import torch
from torch import nn

from lib.lls_solver import lls_solver


# grad_sample: 1/N
def collect_delta(
	layer,
	grad_sample,
	net_state,
	hm,
	init_damping,
	gm,
	step_no,
	device_name,
):
	updates = {}
	if isinstance(layer, nn.EmbeddingBag) or isinstance(layer, nn.Embedding):
		delta = lls_solver(
			layer.weight.grad.to(device=device_name),
			grad_sample[layer.weight],
			net_state[layer.weight],
			hm,
			gm,
			step_no,
			init_damping,
			device_name,
		)
		updates[layer.weight] = delta
	elif isinstance(layer, nn.BatchNorm2d):
		group_gs = torch.cat(
			[
				gs.view(*gs.size()[:2], -1)
				for p in layer.parameters()
				for gs in [grad_sample[p]]
			],
			dim=-1
		)
		group_g = torch.cat(
			[
				g.view(g.size()[0], -1)
				for p in layer.parameters()
				for g in [p.grad.to(device=device_name)]
			],
			dim=-1
		)
		delta = lls_solver(
			group_g,
			group_gs,
			net_state[next(layer.parameters())],
			hm,
			gm,
			step_no,
			init_damping,
			device_name,
		)
		updates[layer.weight] = delta[:, 0]
		updates[layer.bias] = delta[:, 1]
	# return update_with_fisher_2d_no_coeff( group_gs)[:, 0]
	elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
		if isinstance(layer, nn.Conv2d):
			gs = grad_sample[layer.weight]
			g = layer.weight.grad.to(device=device_name)
			# out_channel, *
			weight_delta = lls_solver(
				# batch_size, out_channel * in_channel, x
				g.view(
					prod(g.size()[:2]),
					-1,
				),
				gs.reshape(
					gs.size()[0],
					prod(gs.size()[1:3]),
					-1,
				),
				net_state[layer.weight],
				hm,
				gm,
				step_no,
				init_damping,
				device_name,
			)
		elif isinstance(layer, nn.Linear):
			gs = grad_sample[layer.weight]
			g = layer.weight.grad.to(device=device_name)
			# batch_size, out_channel_of_last * out_channel, x
			# note: require manual modification of torchvision package
			if hasattr(layer, 'original_size'):
				if layer.in_features / layer.original_size[0] > 16:
					weight_delta = lls_solver(
						g.reshape(
							-1,
							1,
						),
						gs.reshape(
							gs.size()[0],
							-1,
							1,
						),
						net_state[layer.weight],
						hm,
						gm,
						step_no,
						init_damping,
						device_name,
					)
				else:
					weight_delta = lls_solver(
						g.reshape(
							layer.out_features * layer.original_size[0],
							-1,
						),
						gs.reshape(
							gs.size()[0],
							layer.out_features * layer.original_size[0],
							-1,
						),
						net_state[layer.weight],
						hm,
						gm,
						step_no,
						init_damping,
						device_name,
					)
			else:
				if layer.in_features > 16:
					weight_delta = lls_solver(
						g.reshape(
							-1,
							1,
						),
						gs.reshape(
							gs.size()[0],
							-1,
							1,
						),
						net_state[layer.weight],
						hm,
						gm,
						step_no,
						init_damping,
						device_name,
					)
				else:
					weight_delta = lls_solver(
						# batch_size, out_channel_of_last * out_channel, x
						g.reshape(
							layer.out_features,
							-1,
						),
						gs.reshape(
							gs.size()[0],
							layer.out_features,
							-1,
						),
						net_state[layer.weight],
						hm,
						gm,
						step_no,
						init_damping,
						device_name,
					)

		updates[layer.weight] = weight_delta.view(*layer.weight.size())

		if layer.bias is not None:
			gs = grad_sample[layer.bias]
			g = layer.bias.grad.to(device=device_name)
			bias_delta = lls_solver(
				g.view(1, -1),
				gs.view(gs.size()[0], 1, -1),
				net_state[layer.bias],
				hm,
				gm,
				step_no,
				init_damping,
				device_name,
			)
			bias_delta = bias_delta[0]
			updates[layer.bias] = bias_delta
	else:
		raise NotImplementedError(f'layer {layer} not supported')
	return updates
