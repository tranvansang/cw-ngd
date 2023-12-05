import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torchvision import models as torchvision_models

from config import learning_rate, optimizer_name, seed_no, total_epoch_num, \
	dataset_name, \
	global_batch_size, comp_size, cw_ngd_damping, gradient_momentum, hessian_momentum, default_momentum, max_acc, \
	max_steps, output_path_prefix, print_all_config, use_multiple_gpus
from lib import model_resnet_for_cifar
from lib.cwngd import CWNGDOptimizer
from lib.data import load_data
from lib.fit import fit
from lib.kfac import KFACOptimizer
from lib.manual_adam import ManualAdam
from lib.manual_sgd import ManualSGD
from lib.model_fmnist_linear import FashionMnistLinear
from lib.model_mnist_conv import MnistConv
from lib.utils import cuda_device_name


def main_single_train(get_max_steps=None, test_every_step=True, fout=sys.stdout):
	# https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
	# torch.use_deterministic_algorithms(True)
	random.seed(seed_no)
	np.random.seed(seed_no)
	torch.manual_seed(seed_no)
	torch.cuda.manual_seed_all(seed_no)

	print_all_config(fout)

	# model
	net = FashionMnistLinear() if dataset_name == 'fmnist' \
		else MnistConv() if dataset_name == 'mnist' \
		else torchvision_models.resnet18(num_classes=10) if dataset_name == 'stl10' \
		else model_resnet_for_cifar.resnet56(10) if dataset_name == 'cifar10' \
		else None
	if net is None: raise Exception(f'Unknown dataset {dataset_name}')
	net = net.to(cuda_device_name)
	# torch.func.replace_all_batch_norm_modules_(net)

	# data
	train_loader, test_loader = load_data(
		name=dataset_name,
		batch_size=global_batch_size,
	)

	# loss function
	criterion = nn.CrossEntropyLoss()

	# optimizer
	optimizer = ManualSGD(
		[p for p in net.parameters() if p.requires_grad],
		lr=.1,
		momentum=.9,
		weight_decay=0,
	) if optimizer_name == 'msgd' \
		else ManualAdam(
		[p for p in net.parameters() if p.requires_grad],
		lr=.001,
		betas=(.9, .99),
		eps=1e-8,
		weight_decay=0,
	) if optimizer_name == 'adam' \
		else KFACOptimizer(net) if optimizer_name == 'kfac' \
		else CWNGDOptimizer(
		[m for m in net.modules()],
		lr=learning_rate,
		momentum=default_momentum,
		gradient_momentum=gradient_momentum,
		hessian_momentum=hessian_momentum,
		comp_size=comp_size,
		damping=cw_ngd_damping,
		multiple_gpus=use_multiple_gpus,
	) if optimizer_name == 'cwngd' \
		else None
	if optimizer is None: raise Exception(f'optimizer {optimizer_name} is not supported')

	# fit
	path_prefix = os.path.join(os.path.dirname(__file__), 'logs', f'stat-{output_path_prefix}')
	with open(f'{path_prefix}train.csv', 'w') as ftrain, \
		open(f'{path_prefix}step.csv', 'w') as fstep, \
		open(f'{path_prefix}test.csv', 'w') as ftest:
		fout.write(f'train log: {path_prefix}train.csv\n')
		fout.write(f'step log: {path_prefix}step.csv\n')
		fout.write(f'test log: {path_prefix}test.csv\n')
		fout.flush()
		return fit(
			fout, ftrain, fstep, ftest, test_every_step,
			net, train_loader, test_loader, criterion, optimizer,
			get_max_steps, epoch_num=total_epoch_num, max_acc=max_acc, max_steps=max_steps,
		)


if __name__ == '__main__':
	start = datetime.now()
	print(start, flush=True)
	try:
		main_single_train()
	finally:
		print(f'total time: {datetime.now() - start} seconds')
