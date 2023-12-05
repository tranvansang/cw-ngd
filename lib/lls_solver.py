import torch

enable_torch_compile = False
# enable_torch_compile = True
_torch_compile = torch.compile if enable_torch_compile else lambda x: x


@_torch_compile
def _lls_tf_cholesky(
	# eq_count, x, x
	h,
	# eq_count, x
	g,
):
	# eq_count, x
	return torch.squeeze(
		# eq_count, x, 1
		torch.cholesky_solve(
			# eq_count, x, 1
			g.unsqueeze(-1),
			# eq_count, x, x
			torch.linalg.cholesky(h),
		),
		dim=-1
	)


# batch_size, eq_count, x
# derivative: scale 1/N
def lls_solver(
	# eq_count, x
	g,
	# batch_size, eq_count, x
	gs,  # gradient per sample
	ps,
	hm,
	gm,
	step_no,
	init_damping,
	device_name,
):
	# eq_count, x, x
	# eq_count, batch_size, x
	# eq_count, x, batch_size
	h = gs.permute(1, 2, 0) @ gs.permute(1, 0, 2)

	bs, eq_count, mat_size = gs.size()
	h /= bs

	h.mul_(1 - hm)
	if 'h_buffer' in ps:
		h.add_(ps['h_buffer'], alpha=hm)
	ps['h_buffer'] = h.clone()  # still detached
	h /= (1 - hm ** step_no)
	# add damping
	h += torch.diag(torch.full((mat_size,), init_damping, device=device_name, requires_grad=False)).unsqueeze(0)

	g = g.clone()
	if 'g_buffer' in ps:
		ps['g_buffer'].mul_(gm).add_(g, alpha=1 - gm)
		g = ps['g_buffer'].clone()
	else:
		g.mul_(1 - gm)
		ps['g_buffer'] = g.clone()
	g /= (1 - gm ** step_no)

	return _lls_tf_cholesky(
		# eq_count, x, x
		h,
		# 1, eq_count, x
		g,
	)
