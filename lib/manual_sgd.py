import torch
import torch.optim as optim


class ManualSGD(optim.Optimizer):
	def __init__(
		self,
		params,
		lr=0.1,
		momentum=0,
		weight_decay=0,
	):
		super(ManualSGD, self).__init__(
			params,
			dict(
				lr=lr,
				momentum=momentum,
				weight_decay=weight_decay,
			)
		)

	def step(self):
		with torch.no_grad():
			for group in self.param_groups:
				um = group['momentum']
				lr = group['lr']
				for p in group['params']:
					if p.grad is None:
						continue
					ps = self.state[p]
					u = p.grad
					if um != 0:
						if 'momentum_buffer' not in ps:
							ps['momentum_buffer'] = u = p.grad.detach().clone()
						else:
							u = ps['momentum_buffer'].mul_(um).add_(u)
					p.add_(u, alpha=-lr)
