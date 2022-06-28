import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):

	def __init__(self, num_classes=2, use_zero_vectors=True, h1=64, h2=64):
		super(Model, self).__init__()

		self._num_classes = num_classes
		self._use_zero_vectors = use_zero_vectors
		self._h1 = h1
		self._h2 = h2

		if self._use_zero_vectors:
			self._in_features = 4
		else:
			self._in_features = 3

		self._mlp = nn.Sequential(
									nn.Linear(self._in_features, self._h1),
									nn.ReLU(),
									nn.Dropout(0.5),
									nn.Linear(self._h1, self._h2),
									nn.ReLU(),
									nn.Dropout(0.5),
									nn.Linear(self._h2, self._num_classes)
								)


		# initialize weights
		for m in self.modules():
			if isinstance(m, (nn.Linear)):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		out = self._mlp(x)

		return out


