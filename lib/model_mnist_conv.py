import torch
import torch.nn.functional as F
from torch import nn


# https://github.com/pytorch/examples/blob/7f7c222b355abd19ba03a7d4ba90f1092973cdbc/mnist/main.py

class MnistConv(nn.Module):
	def __init__(self):
		super(MnistConv, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		if not hasattr(self.fc1, 'original_size'):
			self.fc1.original_size = x.size()[1:]
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output
