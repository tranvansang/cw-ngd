import random

import torch
import torch.optim as optim
from torch.nn import Linear, Conv1d, Conv2d, Conv3d, BatchNorm2d

from lib.collect_delta import collect_delta
from lib.collect_delta_scalar import collect_delta_scalar, collect_update
from lib.hook import make_per_sample_grad_layer_pure, has_trainable_params
from lib.lls_solver import lls_solver
from lib.utils import cuda_device_name

cwngd_known_modules = {
	Conv1d, Conv2d, Conv3d,
	Linear,
}
cwngd_skipped_modules = {BatchNorm2d}


class CWNGDOptimizer(optim.Optimizer):
	def __init__(self, modules, lr, momentum, gradient_momentum, hessian_momentum, damping, comp_size, multiple_gpus):
		self.enable_hooks = True
		self.grad_sampler = {}
		self.step_no = 0
		self.comp_size = abs(comp_size)

		self.inputs = {}
		self.grad_sample = {}

		self.multiple_gpus = multiple_gpus
		if self.multiple_gpus:
			current_device = 0
			gpu_count = torch.cuda.device_count()

			def get_device():
				nonlocal gpu_count
				if gpu_count == 1:
					return cuda_device_name
				nonlocal current_device
				# round-robin
				device = f'cuda:{current_device + 1}'
				current_device = (current_device + 1) % (gpu_count - 1)
				return device
			self.devices = {module: get_device() for module in modules}
			self.streams = {
				module: torch.cuda.Stream(device=self.devices[module]) for module in modules
			}

		if self.comp_size > 0:
			self.param_count = 0
			self.param_offset = {}
			self.comp_state = {}
			self.remainder_comp_state = {}

		# prepare
		self.modules = []
		for module in modules:
			if has_trainable_params(module):
				if any([isinstance(module, cls) for cls in cwngd_known_modules]):
					self.grad_sampler[module] = make_per_sample_grad_layer_pure(module)
					self.modules.append(module)
					module.register_forward_hook(self.forward_hook)
					module.register_full_backward_hook(self.backward_hook)

					if self.comp_size > 0:
						for p in module.parameters(recurse=False):
							if p.requires_grad:
								self.param_offset[p] = self.param_count
								self.param_count += p.numel()
				elif any([isinstance(module, cls) for cls in cwngd_skipped_modules]):
					pass
				else:
					raise NotImplementedError('Unknown module type: %s' % module.__class__.__name__)
		if self.comp_size > 0:
			self.params_pos = list(range(self.param_count))
			if comp_size < 0:
				random.shuffle(self.params_pos)

		super(CWNGDOptimizer, self).__init__(
			(p for m in modules for p in m.parameters(recurse=False) if p.requires_grad),
			dict(
				lr=lr,
				update_momentum=momentum,
				hessian_momentum=hessian_momentum,
				damping=damping,
				gradient_momentum=gradient_momentum,
			)
		)

	def zero_grad(self, *args, **kwargs):
		self.grad_sample = {}
		self.inputs = {}
		super(CWNGDOptimizer, self).zero_grad(*args, **kwargs)

	def forward_hook(self, module, inputs, _forward_output):
		if not self.enable_hooks: return
		if module not in self.inputs:
			self.inputs[module] = []
		if self.multiple_gpus:
			self.inputs[module].append([inp.to(self.devices[module], non_blocking=True) for inp in inputs])
		else:
			self.inputs[module].append(inputs)

	# hook
	def backward_hook(self, module, grad_input, grad_output):
		if not self.enable_hooks: return
		with torch.no_grad():
			if self.multiple_gpus:
				stream = self.streams[module]
				stream.wait_stream(torch.cuda.current_stream())
				with torch.cuda.stream(stream):
					for backprops in grad_output:
						for param, sample_grad in self.grad_sampler[module](
							self.inputs[module][-1],
							backprops.to(self.devices[module], non_blocking=True),
						).items():
							if param in self.grad_sample:
								raise NotImplementedError('Not implemented yet')
								self.grad_sample[param] = self.grad_sample[param] + sample_grad
							else:
								self.grad_sample[param] = sample_grad
			else:
				for backprops in grad_output:
					for param, sample_grad in self.grad_sampler[module](self.inputs[module][-1], backprops).items():
						if param in self.grad_sample:
							raise NotImplementedError('Not implemented yet')
							self.grad_sample[param] = self.grad_sample[param] + sample_grad
						else:
							self.grad_sample[param] = sample_grad
			self.inputs[module].pop()

	def step(self):
		self.step_no += 1

		for m in self.modules:
			for p in m.parameters():
				if p in self.grad_sample:
					if self.multiple_gpus:
						with torch.cuda.stream(self.streams[m]):
							self.grad_sample[p].mul_(self.grad_sample[p].size()[0])
					else:
						self.grad_sample[p].mul_(self.grad_sample[p].size()[0])

		group = self.param_groups[0]
		lr = group['lr']
		hm = group['hessian_momentum']
		init_damping = group['damping']
		gm = group['gradient_momentum']
		um = group['update_momentum']

		for m in self.modules:
			assert len(
				self.inputs[m]) == 0, f'Some inputs are not consumed. {m.__class__.__name__}. left: {len(self.inputs[m])}'
		# https://github.com/tensorflow/kfac/blob/ddad6375bbdebfae809bccfd3a5c3db073128764/kfac/python/ops/optimizer.py#L1327

		# zero out grad: no need to zero out grad because we never retain_grad()

		# real mutable update
		updates = {}
		if self.comp_size > 0:
			if self.multiple_gpus:
				raise NotImplementedError('not implemented')
			# bs x ncomp x self.comp_size
			ncomp = self.param_count // self.comp_size
			remainder_size = self.param_count % self.comp_size

			bs = self.grad_sample[list(self.grad_sample.keys())[0]].size()[0]
			# ncomp x self.comp_size x bs
			group_gs = torch.empty((bs, self.param_count), device=cuda_device_name, dtype=torch.float32)
			# ncomp x self.comp_size
			group_g = torch.empty(self.param_count, device=cuda_device_name, dtype=torch.float32)

			for m in self.modules:
				for p in m.parameters(recurse=False):
					if p.requires_grad:
						offset = self.param_offset[p]
						gs = self.grad_sample[p]
						pos_indices = self.params_pos[offset:offset + p.numel()]

						group_g[pos_indices] = p.grad.view(-1)
						group_gs[:, pos_indices] = gs.reshape(gs.size()[0], -1).to(cuda_device_name)

			# ncomp x self.comp_size
			all_update = lls_solver(
				group_g[:ncomp * self.comp_size].view(ncomp, self.comp_size),
				group_gs[:, :ncomp * self.comp_size].view(bs, ncomp, self.comp_size),
				self.comp_state,
				hm,
				gm,
				self.step_no,
				init_damping,
				cuda_device_name,
			).view(-1)
			if remainder_size > 0:
				remainder_update = lls_solver(
					group_g[ncomp * self.comp_size:].view(1, -1),
					group_gs[:, ncomp * self.comp_size:].view(bs, 1, -1),
					self.remainder_comp_state,
					hm,
					gm,
					self.step_no,
					init_damping,
					cuda_device_name,
				).view(-1)
				all_update = torch.cat([all_update, remainder_update])
			for m in self.modules:
				for p in m.parameters(recurse=False):
					if p.requires_grad:
						offset = self.param_offset[p]
						pos_indices = self.params_pos[offset:offset + p.numel()]
						updates[p] = all_update[pos_indices].view(p.size())
		else:
			for m in self.modules:
				if self.multiple_gpus:
					self.streams[m].wait_stream(torch.cuda.current_stream())
					with torch.cuda.stream(self.streams[m]):
						layer_updates = collect_delta(
							m,
							self.grad_sample,
							self.state,
							hm,
							init_damping,
							gm,
							self.step_no,
							self.devices[m],
						)
				else:
					layer_updates = collect_delta(
						m,
						self.grad_sample,
						self.state,
						hm,
						init_damping,
						gm,
						self.step_no,
						cuda_device_name,
					)
				for p, u in layer_updates.items():
					updates[p] = u
		for group in self.param_groups:
			for p in group['params']:
				if p not in updates:
					ps = self.state[p]
					# note: we can take average of updates here
					updates[p] = collect_delta_scalar(
						p,
						ps,
						hm,
						gm,
						self.step_no,
						init_damping,
					)
		if self.multiple_gpus:
			for m in self.modules:
				torch.cuda.current_stream().wait_stream(self.streams[m])
		for group in self.param_groups:
			for p in group['params']:
				u = collect_update(
					self.state[p],
					updates[p].to(cuda_device_name),
					um,
				)
				p.add_(u, alpha=-lr)
