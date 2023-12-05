import torch


def collect_delta_scalar(
	p,
	ps,
	hm,
	gm,
	step_no,
	init_damping,
):
	g = p.grad
	g = g.clone()

	h = g ** 2
	h.mul_(1 - hm)
	if 'h_buffer' in ps:
		h.add_(ps['h_buffer'], alpha=hm)
	ps['h_buffer'] = h.clone()
	h /= (1 - hm ** step_no)

	if 'damping' not in ps:
		ps['damping'] = torch.ones_like(p, requires_grad=False) * init_damping
	h.sqrt_().add(ps['damping'])

	if 'g_buffer' in ps:
		ps['g_buffer'].mul_(gm).add_(g, alpha=1 - gm)
		g = ps['g_buffer'].clone()
	else:
		g = g.clone().mul_(1 - gm)
		ps['g_buffer'] = g.clone()
	g /= (1 - gm ** step_no)

	g /= h
	return g


def collect_update(
	ps,
	u,  # independent mem
	um,
):
	if um != 0:
		if 'update_buffer' in ps:
			u = u + ps['update_buffer'] * um
		ps['update_buffer'] = u.clone()
	return u
