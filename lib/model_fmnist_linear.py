# https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4

import torch.nn.functional as F
from torch import nn


class FashionMnistLinear(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 120)
		self.fc2 = nn.Linear(120, 120)
		self.fc3 = nn.Linear(120, 10)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = x.view(x.shape[0], -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = F.log_softmax(self.fc3(x), dim=1)
		return x
