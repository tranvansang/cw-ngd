import torch

from config import optimizer_name
from lib.utils import cuda_device_name


def net_acc(net, outputs, targets, step_loss, stat_list):
	# overall accuracy
	total = torch.IntTensor([targets.size(0)]).to(cuda_device_name)
	predicted = torch.argmax(outputs, 1)
	correct = torch.sum(predicted == targets)

	for stat in stat_list:
		if type(step_loss) is dict:
			for k, v in step_loss.items():
				stat.update(k, v * total)
		else:
			stat.update('loss', step_loss * total)

		stat.update('total', total)
		stat.update('correct', correct)


def train(epoch_no, epoch_stat, step_stat, net, inputs, targets, optimizer, criterion):
	net.train()
	if optimizer_name == 'cwngd': optimizer.enable_hooks = True
	inputs, targets = inputs.to(cuda_device_name), targets.to(cuda_device_name)

	# reset the gradients
	for param in net.parameters():
		if param.grad is not None:
			param.grad.zero_()

	# train
	# forward + backward
	optimizer.zero_grad()
	outputs = net(inputs)
	step_loss = criterion(outputs, targets)

	if optimizer_name == 'kfac' and optimizer.steps % optimizer.TCov == 0:
		# compute true fisher
		optimizer.acc_stats = True
		with torch.no_grad():
			sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1), 1).squeeze()
		loss_sample = criterion(outputs, sampled_y)
		loss_sample.backward(retain_graph=True)
		optimizer.acc_stats = False
		optimizer.zero_grad()  # clear the gradient for computing true-fisher.

	if torch.isnan(step_loss):
		raise ValueError('NaN loss')
	step_loss.backward()

	if optimizer_name == 'cwngd': optimizer.enable_hooks = False
	for n in (net if type(net) in (list, tuple) else (net,)):
		n.eval()

	with torch.no_grad():
		optimizer.step()
		net_acc(net, outputs, targets, step_loss, [step_stat, epoch_stat])
