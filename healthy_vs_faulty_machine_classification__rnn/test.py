import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset_test import Dataset

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')

parser.add_argument('--init_model_file', default='', help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--data_dir', default='../data', help='Data folder', dest='data_dir')
parser.add_argument('--train_dataset_file', default='../dataset/train_filelist.txt', help='train dataset file', dest='train_dataset_file')
parser.add_argument('--val_dataset_file', default='../dataset/train_filelist.txt', help='validation dataset file', dest='val_dataset_file')
parser.add_argument('--test_dataset_file', default='../dataset/test_filelist.txt', help='test dataset file', dest='test_dataset_file')
parser.add_argument('--train_val_split_ratio', default='0.7', type=float, help='Ratio to split dataset into training and validation', dest='train_val_split_ratio')
parser.add_argument('--dataset_type', default='test', help='', dest='dataset_type')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--num_hidden_nodes1', default='64', type=int, help='Number of nodes in hidden layer 1', dest='num_hidden_nodes1')
parser.add_argument('--num_hidden_nodes2', default='64', type=int, help='Number of nodes in hidden layer 2', dest='num_hidden_nodes2')
parser.add_argument('--data_length', default='5', type=int, help='Length of data (periods)', dest='data_length')
parser.add_argument('--sampling_frequency', default='40000', type=int, help='Sampling frequency (Hz)', dest='sampling_frequency')
parser.add_argument('--use_zero_vectors', default=True, type=str2bool, help='Flag to control using zero vectors', dest='use_zero_vectors')
parser.add_argument('--batch_size', default='32', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--test_metrics_dir', default='test_metrics', help='Text file to write test metrics', dest='test_metrics_dir')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

print('Parameters:')
for key in FLAGS_dict.keys():
	print('# {} = {}'.format(key, FLAGS_dict[key]))

if FLAGS.dataset_type == 'test':
	dataset_file = FLAGS.test_dataset_file
	split_ratio = None
elif FLAGS.dataset_type == 'val':
	dataset_file = FLAGS.val_dataset_file
	split_ratio = FLAGS.train_val_split_ratio
elif FLAGS.dataset_type == 'train':
	dataset_file = FLAGS.train_dataset_file	
	split_ratio = FLAGS.train_val_split_ratio

print('Preparing dataset ...')
dataset = Dataset(	data_dir=FLAGS.data_dir, 
					dataset_file=dataset_file, 
					dataset_type=FLAGS.dataset_type, 
					data_length=FLAGS.data_length,
					sampling_frequency=FLAGS.sampling_frequency,
					use_zero_vectors=FLAGS.use_zero_vectors,
					split_ratio = split_ratio,
					balanced_dataset = False)
num_samples = dataset.num_samples
print("Data - num_samples: {}".format(num_samples))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1)

model = Model(num_classes=FLAGS.num_classes, use_zero_vectors=FLAGS.use_zero_vectors, h1=FLAGS.num_hidden_nodes1, h2=FLAGS.num_hidden_nodes2)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file)
		model.load_state_dict(state_dict['model_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))


model_name = FLAGS.init_model_file.split('__')[1] + '__' + FLAGS.init_model_file.split('__')[2] + '__' + FLAGS.init_model_file.split('__')[3][:-4]
data_folder_path = '{}/{}/{}'.format(FLAGS.test_metrics_dir,model_name,FLAGS.dataset_type)
if not os.path.exists(data_folder_path):
	os.makedirs(data_folder_path)


stat_dict = dict()
truth_list = []
pred_list = []
probs_list = []

model.eval()
with torch.no_grad():

	pbar = tqdm(total=len(data_loader))
	for (file_ids,point_ids), samples, targets in data_loader:
		# print(sample_ids)
		samples = samples.to(device)
		targets = targets.to(device)

		# get logits from model
		batch_logits = model(samples)
		batch_probs = F.softmax(batch_logits, dim=1)
		batch_probs_arr = batch_probs.cpu().numpy()

		num_points = targets.size(0)
		# print('num_points: {}'.format(num_points))

		batch_truths_arr = np.asarray(targets.numpy(),dtype=int)
		batch_preds_arr = np.argmax(batch_probs_arr, axis=1)

		pbar.update(1)


		for n in range(num_points):
			temp_file_id = file_ids[n]
			temp_point_id = point_ids[n]
			temp_truth = batch_truths_arr[n]
			temp_pred = batch_preds_arr[n]
			temp_probs = batch_probs_arr[n]

			if temp_file_id not in stat_dict:
				stat_dict[temp_file_id] = {'point_id':[],'truth':[],'pred':[],'probs':[]}

			stat_dict[temp_file_id]['point_id'].append(temp_point_id)
			stat_dict[temp_file_id]['truth'].append(temp_truth)
			stat_dict[temp_file_id]['pred'].append(temp_pred)
			stat_dict[temp_file_id]['probs'].append(temp_probs)

			truth_list.append(temp_truth)
			pred_list.append(temp_pred)
			probs_list.append(temp_probs)


	pbar.close()


# write to files
num_data_files = len(stat_dict)
for file_ind,file_id in enumerate(sorted(stat_dict.keys())):
	print('File {}/{}:{}'.format(file_ind+1,num_data_files,file_id))

	# create sample data folder
	sample_data_folder_path = '{}/{}'.format(data_folder_path,file_id)
	if not os.path.exists(sample_data_folder_path):
		os.makedirs(sample_data_folder_path)

	# create test metrics file
	test_metrics_filename = '{}/predictions_{}.txt'.format(sample_data_folder_path,file_id)

	with open(test_metrics_filename,'w') as f_metric_file:
		f_metric_file.write('# Parameters:\n')

		for key in FLAGS_dict.keys():
			f_metric_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

		f_metric_file.write('# point_id\ttruth\tpred')
		for c in range(FLAGS.num_classes):
			f_metric_file.write('\tprob_{}'.format(c))
		f_metric_file.write('\n')


	# get data
	point_id_arr = np.array(stat_dict[file_id]['point_id'])
	truth_arr = np.array(stat_dict[file_id]['truth'])
	pred_arr = np.array(stat_dict[file_id]['pred'])
	probs_arr = np.array(stat_dict[file_id]['probs'])

	# write to test metrics file
	for n in range(point_id_arr.shape[0]):
		temp_point_id = point_id_arr[n]
		temp_truth = truth_arr[n]
		temp_pred = pred_arr[n]
		temp_probs = probs_arr[n]

		with open(test_metrics_filename,'a') as f_metric_file:
			f_metric_file.write('{:d}\t{:d}\t{:d}'.format(temp_point_id,temp_truth,temp_pred))
			for c in range(FLAGS.num_classes):
				f_metric_file.write('\t{:.4f}'.format(temp_probs[c]))
			f_metric_file.write('\n')




