import torch

from lib.utils import cuda_device_name
from lib.step_stat import StepStat
from lib.train import train, net_acc


def fit(
	fout, ftrain, fstep, ftest, test_every_step,
	net, train_loader, test_loader, criterion, optimizer,
	get_max_steps, epoch_num, max_acc, max_steps,
	lr_scheduler=None):
	train_stat = StepStat(file=ftrain)
	step_stat = StepStat(file=fstep)
	test_stat = StepStat(file=ftest)

	def do_test():
		with torch.no_grad():
			with test_stat:
				for inputs, targets in test_loader:
					inputs, targets = inputs.to(cuda_device_name), targets.to(cuda_device_name)
					outputs = net(inputs)
					step_loss = criterion(outputs, targets)
					net_acc(net, outputs, targets, step_loss, [test_stat])
		if test_stat.last_acc is not None and 0 < max_acc < test_stat.last_acc:
			fout.write(
				'testing accuracy {test_acc} reached threshold {max_acc} at step {step}, epoch {epoch}, exit now!\n'.format(
					max_acc=max_acc,
					test_acc=test_stat.last_acc,
					step=step_stat.step_no,
					epoch=epoch_no,
				))
			return {
				'reason': 'max_acc',
				'epoch_no': epoch_no,
				'step_no': step_stat.step_no,
				'train_loss': train_stat.last_loss,
				'train_acc': train_stat.last_acc,
				'test_loss': test_stat.last_loss,
				'test_acc': test_stat.last_acc,
			}

	for epoch_no in range(epoch_num):
		with train_stat:
			for inputs, targets in train_loader:
				with step_stat:
					train(epoch_no, train_stat, step_stat, net, inputs, targets, optimizer, criterion)
				if max_steps != -1 and step_stat.step_no >= max_steps \
					or \
					get_max_steps is not None and get_max_steps() is not None and step_stat.step_no >= get_max_steps():
					fout.write('max_steps {max_steps} reached at step {step}, epoch {epoch}, exit now!\n'.format(
						max_steps=max_steps,
						step=step_stat.step_no,
						epoch=epoch_no,
					))
					return {
						'reason': 'out_of_steps',
						'epoch_no': epoch_no,
						'step_no': step_stat.step_no,
						'train_loss': train_stat.last_loss,
						'train_acc': train_stat.last_acc,
						'test_loss': test_stat.last_loss,
						'test_acc': test_stat.last_acc,
					}
				if lr_scheduler is not None:
					lr_scheduler.step()
				if test_every_step:
					test_ret = do_test()
					if test_ret is not None:
						return test_ret
		if not test_every_step:
			test_ret = do_test()
			if test_ret is not None:
				return test_ret

	return {
		'reason': 'out_of_epochs',
		'step_no': step_stat.step_no,
		'train_loss': train_stat.last_loss,
		'train_acc': train_stat.last_acc,
		'test_loss': test_stat.last_loss,
		'test_acc': test_stat.last_acc,
	}
