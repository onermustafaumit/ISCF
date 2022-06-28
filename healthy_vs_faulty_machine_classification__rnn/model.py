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

		self._rnncell = nn.RNNCell(input_size=1, hidden_size=self._h1, nonlinearity='relu', bias=True)
		self._linear = nn.Linear(self._h1, self._num_classes)

	def forward(self, x):
		x = torch.unsqueeze(x, dim=-1)
		N, L, H_in = x.shape

		h = self._rnncell(x[:,0,:])
		h = F.dropout(h, p=0.5, training=self.training)

		for i in range(1,L):
			h = self._rnncell(x[:,i,:],h)
			h = F.dropout(h, p=0.5, training=self.training)

		out = self._linear(h)

		return out





		


