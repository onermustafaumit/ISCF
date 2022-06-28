import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):

	def __init__(self, num_classes=4, use_zero_vectors=True, h1=32, h2=16):
		super(Model, self).__init__()

		self._num_classes = num_classes
		self._use_zero_vectors = use_zero_vectors
		self._h1 = h1
		self._h2 = h2

		if self._use_zero_vectors:
			self._in_features = 4
		else:
			self._in_features = 3

		self._cnn = nn.Sequential(
									nn.Conv1d(in_channels=1, out_channels=self._h1, kernel_size=3, stride=1, padding=1),
									nn.ReLU(),
									nn.Dropout(0.5),
									nn.Conv1d(in_channels=self._h1, out_channels=self._h2, kernel_size=2, stride=1, padding=0),
									nn.ReLU(),
									nn.Dropout(0.5),
									nn.Conv1d(in_channels=self._h2, out_channels=1, kernel_size=2, stride=1, padding=0)
								)

	def forward(self, x):
		x = torch.unsqueeze(x, dim=1)
		x = self._cnn(x)
		out = torch.squeeze(x, dim=1)

		return out





		


