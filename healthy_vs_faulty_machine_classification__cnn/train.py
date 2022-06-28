import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset import Dataset #, custom_collate_fn, worker_init_fn

import torch
import torch.utils.data

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
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--num_hidden_nodes1', default='32', type=int, help='Number of nodes in hidden layer 1', dest='num_hidden_nodes1')
parser.add_argument('--num_hidden_nodes2', default='64', type=int, help='Number of nodes in hidden layer 2', dest='num_hidden_nodes2')
parser.add_argument('--data_length', default='5', type=int, help='Length of data (periods)', dest='data_length')
parser.add_argument('--sampling_frequency', default='40000', type=int, help='Sampling frequency (Hz)', dest='sampling_frequency')
parser.add_argument('--use_zero_vectors', default=True, type=str2bool, help='Flag to control using zero vectors', dest='use_zero_vectors')
parser.add_argument('--balanced_dataset', default=False, type=str2bool, help='Flag to create balanced dataset', dest='balanced_dataset')
parser.add_argument('--batch_size', default='32', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--learning_rate', default='3e-4', type=float, help='Learning rate', dest='learning_rate')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs', dest='num_epochs')
parser.add_argument('--save_interval', default=10, type=int, help='Model save interval (default: 1000)', dest='save_interval')
parser.add_argument('--metrics_file', default='loss_data', help='Text file to write step, loss, accuracy metrics', dest='metrics_file')
parser.add_argument('--model_dir', default='saved_models', help='Directory to save models', dest='model_dir')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

if not os.path.exists(FLAGS.metrics_file):
	os.makedirs(FLAGS.metrics_file)

if not os.path.exists(FLAGS.model_dir):
	os.makedirs(FLAGS.model_dir)

current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
metrics_file = '{}/step_loss_acc_metrics{}.txt'.format(FLAGS.metrics_file, current_time)

print('Model parameters:')
print('data_dir = {}'.format(FLAGS.data_dir))
print('num_classes = {}'.format(FLAGS.num_classes))
print('batch_size = {}'.format(FLAGS.batch_size))
print('learning_rate = {}'.format(FLAGS.learning_rate))
print('num_epochs = {}'.format(FLAGS.num_epochs))
print('metrics_file = {}'.format(FLAGS.metrics_file))

print('Preparing training dataset ...')
train_dataset = Dataset(data_dir=FLAGS.data_dir, 
						dataset_file=FLAGS.train_dataset_file, 
						dataset_type='train', 
						data_length=FLAGS.data_length,
						sampling_frequency=FLAGS.sampling_frequency,
						use_zero_vectors=FLAGS.use_zero_vectors,
						split_ratio = FLAGS.train_val_split_ratio,
						balanced_dataset = FLAGS.balanced_dataset)
num_samples_train = train_dataset.num_samples
print("Training Data - num_samples: {}".format(num_samples_train))

print('Preparing validation dataset ...')
val_dataset = Dataset(	data_dir=FLAGS.data_dir, 
						dataset_file=FLAGS.val_dataset_file, 
						dataset_type='val', 
						data_length=FLAGS.data_length,
						sampling_frequency=FLAGS.sampling_frequency,
						use_zero_vectors=FLAGS.use_zero_vectors,
						split_ratio = FLAGS.train_val_split_ratio,
						balanced_dataset = FLAGS.balanced_dataset)
num_samples_val = val_dataset.num_samples
print("Validation Data - num_samples: {}".format(num_samples_val))

print('Preparing test dataset ...')
test_dataset = Dataset(	data_dir=FLAGS.data_dir, 
						dataset_file=FLAGS.test_dataset_file, 
						dataset_type='test', 
						data_length=FLAGS.data_length,
						sampling_frequency=FLAGS.sampling_frequency,
						use_zero_vectors=FLAGS.use_zero_vectors,
						split_ratio = None,
						balanced_dataset = False)
num_samples_test = test_dataset.num_samples
print("Test Data - num_samples: {}".format(num_samples_test))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1) #, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1) #, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1) #, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

model = Model(num_classes=FLAGS.num_classes, use_zero_vectors=FLAGS.use_zero_vectors, h1=FLAGS.num_hidden_nodes1, h2=FLAGS.num_hidden_nodes2)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# define loss criterion
criterion = torch.nn.CrossEntropyLoss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=0.0005)

if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file)
		model.load_state_dict(state_dict['model_state_dict'])
		# optimizer.load_state_dict(state_dict['optimizer_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))

with open(metrics_file,'w') as f_metric_file:
	f_metric_file.write('# Model parameters:\n')

	for key in FLAGS_dict.keys():
		f_metric_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

	f_metric_file.write("# Training Data - num_samples: {}\n".format(num_samples_train))
	f_metric_file.write("# Validation Data - num_samples: {}\n".format(num_samples_val))
	f_metric_file.write("# Validation Data - num_samples: {}\n".format(num_samples_val))
	
	f_metric_file.write('# epoch\ttraining_acc\ttraining_loss\tvalidation_acc\tvalidation_loss\ttest_acc\ttest_loss\n')


min_validation_loss = 1000
max_validation_acc = 0.0
best_models_file = FLAGS.model_dir + "/best_models" + current_time + ".txt"
with open(best_models_file,'w') as f_best_models_file:
		f_best_models_file.write("# epoch\ttraining_acc\ttraining_loss\tvalidation_acc\tvalidation_loss\ttest_acc\ttest_loss\n")

for epoch in range(FLAGS.num_epochs):
	# print('############## EPOCH - {} ##############'.format(epoch+1))
	training_loss = 0
	validation_loss = 0
	test_loss = 0

	# train for one epoch
	# print('******** training ********')

	num_corrects = 0
	num_predictions = 0

	pbar = tqdm(total=len(train_data_loader))
	
	model.train()
	for samples, targets in train_data_loader:
		# print(samples.size())
		# print(targets.size())
		samples = samples.to(device)
		targets = targets.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		y_logits = model(samples)
		loss = criterion(y_logits, targets)
		loss.backward()
		optimizer.step()

		training_loss += loss.item()*targets.size(0)

		num_predictions += targets.size(0)

		target_labels = targets
		predicted_labels = torch.argmax(y_logits, dim=1)

		correct_predictions = torch.sum(predicted_labels == target_labels)

		num_corrects += correct_predictions.item()

		pbar.update(1)

		# print(loss.item())

	training_loss /= num_predictions

	training_acc = num_corrects / num_predictions

	pbar.close()


	# evaluate on the validation dataset
	# print('******** validation ********')

	num_corrects = 0
	num_predictions = 0

	pbar = tqdm(total=len(val_data_loader))

	model.eval()
	with torch.no_grad():
		for samples, targets in val_data_loader:
			samples = samples.to(device)
			targets = targets.to(device)

			# forward
			y_logits = model(samples)
			loss = criterion(y_logits, targets)

			validation_loss += loss.item()*targets.size(0)

			num_predictions += targets.size(0)
		
			target_labels = targets
			predicted_labels = torch.argmax(y_logits, dim=1)

			correct_predictions = torch.sum(predicted_labels == target_labels)

			num_corrects += correct_predictions.item()

			pbar.update(1)

	validation_loss /= num_predictions

	validation_acc = num_corrects / num_predictions

	pbar.close()


	# evaluate on the test dataset
	# print('******** test ********')

	num_corrects = 0
	num_predictions = 0

	pbar = tqdm(total=len(test_data_loader))

	model.eval()
	with torch.no_grad():
		for samples, targets in test_data_loader:
			samples = samples.to(device)
			targets = targets.to(device)

			# forward
			y_logits = model(samples)
			loss = criterion(y_logits, targets)

			test_loss += loss.item()*targets.size(0)

			num_predictions += targets.size(0)
		
			target_labels = targets
			predicted_labels = torch.argmax(y_logits, dim=1)

			correct_predictions = torch.sum(predicted_labels == target_labels)

			num_corrects += correct_predictions.item()

			pbar.update(1)

	test_loss /= num_predictions

	test_acc = num_corrects / num_predictions

	pbar.close()

	print('Epoch=%d ### training_acc=%5.3f, training_loss=%5.3f ### validation_acc=%5.3f, validation_loss=%5.3f ### test_acc=%5.3f, test_loss=%5.3f' % (epoch+1, training_acc, training_loss, validation_acc, validation_loss, test_acc, test_loss))

	with open(metrics_file,'a') as f_metric_file:
		f_metric_file.write('%d\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n' % (epoch+1, training_acc, training_loss, validation_acc, validation_loss, test_acc, test_loss))


	# save model
	if (epoch+1) % FLAGS.save_interval == 0:
		model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__' + str(epoch+1) + ".pth"
		state_dict = {	'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
		torch.save(state_dict, model_weights_filename)
		print("Model weights saved in file: ", model_weights_filename)


	# save the best model
	if validation_loss < min_validation_loss or validation_acc > max_validation_acc:
			if validation_loss < min_validation_loss:
				min_validation_loss = validation_loss

			if validation_acc > max_validation_acc:
				max_validation_acc = validation_acc

			model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__best_' + str(epoch+1) + ".pth"
			state_dict = {  'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state_dict, model_weights_filename)
			print("Best model weights saved in file: ", model_weights_filename)

			with open(best_models_file,'a') as f_best_models_file:
					f_best_models_file.write('%d\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n' % (epoch+1, training_acc, training_loss, validation_acc, validation_loss, test_acc, test_loss))


print('Training finished!!!')

model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__' + str(epoch+1) + ".pth"
state_dict = {	'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)
