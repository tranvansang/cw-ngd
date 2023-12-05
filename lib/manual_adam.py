from math import sqrt

import torch
import torch.optim as optim


class ManualAdam(optim.Optimizer):
	def __init__(
		self,
		params,
		lr,
		betas,
		eps,
		weight_decay,
	):
		super(ManualAdam, self).__init__(
			params,
			dict(
				lr=lr,
				betas=betas,
				eps=eps,
				weight_decay=weight_decay,
			)
		)
		self.steps = 0

	def step(self):
		with torch.no_grad():
			for group in self.param_groups:
				beta_1, beta_2 = group['betas']
				bias_correction1 = 1 - beta_1 ** (self.steps + 1)
				bias_correction2 = 1 - beta_2 ** (self.steps + 1)
				bias_correction2_sqrt = sqrt(bias_correction2)

				weight_decay = group['weight_decay']
				for p in group['params']:
					if p.grad is None:
						continue
					if weight_decay != 0:
						p.grad.add_(p, alpha=weight_decay)

					p_state = self.state[p]

					# Decay the first and second moment running average coefficient
					if 'exp_avg' not in p_state:
						p_state['exp_avg'] = torch.zeros_like(p)
					p_state['exp_avg'].mul_(beta_1).add_(p.grad, alpha=1 - beta_1)

					if 'exp_avg_sq' not in p_state:
						p_state['exp_avg_sq'] = torch.zeros_like(p)
					p_state['exp_avg_sq'].mul_(beta_2).add_(p.grad ** 2, alpha=1 - beta_2)
					# p_state['exp_avg_sq'].addcmul_(p.grad, p.grad.conj(), value=1 - beta_2)

					p.add_(
						p_state['exp_avg'] / bias_correction1 / (
							torch.sqrt(p_state['exp_avg_sq']) / bias_correction2_sqrt + group['eps']),
						alpha=-group['lr']
					)
		self.steps += 1
