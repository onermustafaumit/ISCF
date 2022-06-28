import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
	def __init__(self, 	data_dir=None, 
						dataset_file=None, 
						dataset_type=None, 
						data_length=1,
						sampling_frequency=40000,
						use_zero_vectors=True,
						split_ratio = None,
						balanced_dataset=False):

		self._data_dir = data_dir
		self._dataset_file = dataset_file
		self._dataset_type = dataset_type
		self._data_length = data_length 
		self._sampling_frequency = sampling_frequency 
		self._use_zero_vectors = use_zero_vectors 
		self._split_ratio = split_ratio 
		self._balanced_dataset = balanced_dataset

		if self._use_zero_vectors:
			self._bin_edges = np.arange(5) - 0.5
		else:
			self._bin_edges = np.arange(1,5) - 0.5


		# load data
		self._samples_arr, self._labels_arr = self.prepare_data()

		self._indices = np.arange(self._labels_arr.shape[0])

		##### make a balanced dataset #####
		if self._balanced_dataset:
		
			indices_list = list()

			# get unique label ids and counts
			unique_ids, unique_id_counts = np.unique(self._labels_arr[:,0], return_counts=True)
			print('before balancing')
			print(unique_ids, unique_id_counts)

			max_count = np.amax(unique_id_counts)

			for i in range(unique_ids.shape[0]):
				temp_unique_id = unique_ids[i]
				temp_unique_id_count = unique_id_counts[i]

				temp_indices = np.where(self._labels_arr[:,0] == temp_unique_id)[0]
				if temp_unique_id_count < max_count:
					temp_indices = np.tile(temp_indices, int(max_count//temp_unique_id_count + 1) )
					temp_indices = temp_indices[:max_count]

				indices_list += list(temp_indices)

			# print('len(indices_list):{}'.format(len(indices_list)))
			indices_arr = np.array(indices_list)

			self._indices = self._indices[indices_arr]

			unique_ids, unique_id_counts = np.unique(self._labels_arr[self._indices,0], return_counts=True)
			print('after balancing')
			print(unique_ids, unique_id_counts)
			
		##### make a balanced dataset #####

		self._num_samples = self._indices.shape[0]



	@property
	def num_samples(self):
		return self._num_samples

	def __len__(self):
		return self._num_samples


	def prepare_data(self):
		info_arr = np.loadtxt(self._dataset_file, delimiter='\t', comments='#', dtype='str') #[:2,:]
		num_data_files = info_arr.shape[0]

		sample_list = []
		label_list = []
		for i in range(num_data_files):
			file_id = info_arr[i,0]
			n = int(info_arr[i,1])
			w = int(info_arr[i,2])
			T = int(info_arr[i,3])
			f_e = float(info_arr[i,4])
			# MS = int(info_arr[i,5])
			# PS_A = int(info_arr[i,6])
			# PS_B = int(info_arr[i,7])
			# PS_C = int(info_arr[i,8])
			temp_label = np.asarray(info_arr[i,5:],dtype=np.uint8)
			
			# num vectors to be used in histogram calculation
			N = self._data_length*int(self._sampling_frequency / f_e)

			# read vector data
			vector_data_arr_file = '{}/{}.txt'.format(self._data_dir,file_id)
			vector_data_arr = np.loadtxt(vector_data_arr_file, delimiter='\t', comments='#', dtype=int)
			# print('before split - vector_data_arr.shape:{}'.format(vector_data_arr.shape))

			num_vectors = vector_data_arr.shape[0]

			# split the data
			if self._split_ratio != None:
				ind = int(num_vectors*self._split_ratio)
				if self._dataset_type == 'train':
					vector_data_arr = vector_data_arr[:ind]
				else:
					vector_data_arr = vector_data_arr[ind:]

			# print('after split - vector_data_arr.shape:{}'.format(vector_data_arr.shape))


			# group control vectors: v1-v4, v2-v5, v3-v6
			vector_data_arr[vector_data_arr==4]=1
			vector_data_arr[vector_data_arr==5]=2
			vector_data_arr[vector_data_arr==6]=3

			# prepare vector data histograms
			num_vectors = vector_data_arr.shape[0]

			num_samples = num_vectors - N + 1
			for j in range(num_samples):
				vector_data = vector_data_arr[j:j+N]
				vector_counts, _ = np.histogram(vector_data, bins=self._bin_edges, density=True)
				
				sample_list.append(vector_counts)
				label_list.append(temp_label)

		# return samples and labels
		return np.array(sample_list, dtype=np.float32), np.array(label_list, dtype=np.uint8)


	def __getitem__(self, idx):

		temp_index = self._indices[idx]

		temp_sample = self._samples_arr[temp_index]
		temp_label = self._labels_arr[temp_index][0]

		temp_sample = torch.as_tensor(temp_sample, dtype=torch.float32)

		temp_label = torch.as_tensor(temp_label, dtype=torch.long)

		return temp_sample, temp_label


# def custom_collate_fn(batch):
# 	sample_tensors_list, label_tensors_list = zip(*batch)

# 	return torch.stack(sample_tensors_list,dim=0), torch.stack(label_tensors_list,dim=0)

# def worker_init_fn(id):
# 	np.random.seed(torch.initial_seed()&0xffffffff)

