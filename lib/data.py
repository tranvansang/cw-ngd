import os

import torch
import torchvision
from torch.utils.data import Sampler
from torchvision import transforms

# https://github.com/pytorch/pytorch/issues/49180
# default False
_drop_remainder_in_train = False
_root_dir = os.path.join(os.path.dirname(__file__), '..')


def load_data(
	name,
	batch_size,
	data_dir=f'{_root_dir}/data',
	to_download=False,
):
	collate_batch = None

	if name == 'fmnist':
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,)),
		])
		train_dataset = torchvision.datasets.FashionMNIST(
			root=f'{data_dir}/F_MNIST',
			train=True,
			download=to_download,
			transform=transform,
		)

		test_dataset = torchvision.datasets.FashionMNIST(
			root=f'{data_dir}/F_MNIST',
			train=False,
			download=to_download,
			transform=transform,
		)
	elif name == 'mnist':
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
		])
		train_dataset = torchvision.datasets.MNIST(
			root=f'{data_dir}/MNIST',
			train=True,
			download=to_download,
			transform=transform,
		)

		test_dataset = torchvision.datasets.MNIST(
			root=f'{data_dir}/MNIST',
			train=False,
			download=to_download,
			transform=transform,
		)
	elif name == 'cifar10':
		normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		train_dataset = torchvision.datasets.CIFAR10(
			root=f'{data_dir}/CIFAR10',
			train=True,
			download=to_download,
			transform=transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]),
		)

		test_dataset = torchvision.datasets.CIFAR10(
			root=f'{data_dir}/CIFAR10',
			train=False,
			download=to_download,
			transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			]),
		)
	elif name == 'stl10':
		normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
		train_dataset = torchvision.datasets.STL10(
			root=f'{data_dir}/STL10',
			split='train',
			download=to_download,
			transform=transforms.Compose([
				transforms.Resize((272, 272)),
				transforms.RandomRotation(15, ),
				transforms.RandomCrop(256),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]),
		)
		test_dataset = torchvision.datasets.STL10(
			root=f'{data_dir}/STL10',
			split='test',
			download=to_download,
			transform=transforms.Compose([
				transforms.Resize(256),
				transforms.ToTensor(),
				normalize,
			]),
		)
	else:
		raise NotImplementedError(f'Unknown dataset: {name}')

	train_sampler = None
	test_sampler = None
	batch_size_per_worker = batch_size

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size_per_worker,
		shuffle=train_sampler is None,
		sampler=train_sampler,
		drop_last=_drop_remainder_in_train,
		num_workers=0,
		collate_fn=collate_batch,
	)

	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=batch_size_per_worker,
		shuffle=False,
		sampler=test_sampler,
		num_workers=0,
		collate_fn=collate_batch,
	) if test_dataset is not None else None

	return train_loader, test_loader


if __name__ == '__main__':
	for name in ('mnist', 'fmnist', 'cifar10', 'stl10'):
		print('Download dataset, if not exist: ', name)
		load_data(
			name=name,
			batch_size=32,
			to_download=True,
		)
