import numpy as np
import argparse
import os
import sys
from os import path
import itertools
from itertools import cycle

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_fscore_support

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats

plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})


def score_fnc(y_true, y_score):
	auc = roc_auc_score(y_true, y_score)
	return auc

def BootStrap(y_true, y_score, n_bootstraps):

	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []
	# print(y_true)
	# print(y_score)

	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(y_score), len(y_score))

		if len(np.unique(y_score[indices])) < 2:
			# We need at least one sample from each class
			# otherwise reject the sample
			#print("We need at least one sample from each class")
			continue
		else:
			score = score_fnc(y_true[indices], y_score[indices])
			bootstrapped_scores.append(score)
			# print("score: %f" % score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	# Computing the lower and upper bound of the 95% confidence interval
	# You can change the bounds percentiles to 0.025 and 0.975 to get
	# a 95% confidence interval instead.
	#print(sorted_scores)
	#print(len(sorted_scores))
	#print(int(0.025 * len(sorted_scores)))
	#print(int(0.975 * len(sorted_scores)))
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	# print(confidence_lower)
	# print(confidence_upper)
	# print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
	return sorted_scores, confidence_lower, confidence_upper


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues,
						  current_ax = None):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		# print("Normalized confusion matrix")
	else:
		# print('Confusion matrix, without normalization')
		pass

	cm_normalized = (cm.astype('float') - np.amin(cm)) / (np.amax(cm)-np.amin(cm))

	# print(cm)

	plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})
	ax = current_ax
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	# plt.title(title)
	# plt.colorbar()
	tick_marks = np.arange(len(classes))
	ax.set_xticks(tick_marks)
	ax.set_yticks(tick_marks)
	ax.set_xticklabels(classes)
	ax.set_yticklabels(classes)
	ax.set_ylim( (len(classes)-0.5, -0.5) )


	fmt = '.3f' if normalize else 'd'
	thresh = 0.5
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",verticalalignment="center",
				 fontsize=10, fontname='Times New Roman',
				 color="white" if cm_normalized[i, j] > thresh else "black")

	ax.set_ylabel('Truth')
	ax.set_xlabel('Predicted')
	
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)

	# plt.colorbar(im, cax=cax)
	# plt.tight_layout()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='data folder path', dest='data_folder_path')
FLAGS = parser.parse_args()

class_names = ['-ve','+ve']

data_folder_path = FLAGS.data_folder_path

data_series_list = [d for d in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, d))]

data_series_arr = np.asarray(sorted(data_series_list))
num_data_series = data_series_arr.shape[0]
print("Data - num_data_series: {}".format(num_data_series))


data_series_predictions_file = '{}/data_series_predictions.txt'.format(data_folder_path)
with open(data_series_predictions_file,'w') as f_data_series_predictions_file:
	f_data_series_predictions_file.write('# data_series_id\t')
	f_data_series_predictions_file.write('truth\t')
	f_data_series_predictions_file.write('predicted\t')
	f_data_series_predictions_file.write('prob_0\t')
	f_data_series_predictions_file.write('prob_1')
	f_data_series_predictions_file.write('\n')


sample_ids_list = []
sample_truths_list = []
sample_preds_list = []
sample_probs_list = []
data_series_truths_list = []
data_series_preds_list = []
data_series_probs_list = []
for i in range(num_data_series):

	data_series_id = data_series_arr[i]
	print('image {}/{}: {}'.format(i+1,num_data_series,data_series_id))

	# if data_series_id != 'data_041':
	# 	continue

	data_series_folder_path = '{}/{}'.format(data_folder_path,data_series_id)

	test_metrics_filename = '{}/predictions_{}.txt'.format(data_series_folder_path,data_series_id)

	test_metrics_data = np.loadtxt(test_metrics_filename, delimiter='\t', comments='#', dtype=str)
	sample_ids_data = np.asarray(test_metrics_data[:,0],dtype=int)
	truths_data = np.asarray(test_metrics_data[:,1],dtype=int)
	preds_data = np.asarray(test_metrics_data[:,2],dtype=int)
	probs_data = np.asarray(test_metrics_data[:,3:],dtype=float)

	fig, ax = plt.subplots(figsize=(2.3,1.6))
	ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

	# ax.axhline(y=0.5, linestyle='--', linewidth=1, color='red', alpha=0.5)
	ax.plot(sample_ids_data, probs_data[:,1], lw=1, alpha=.8, color='k')
	ax2.step(sample_ids_data, preds_data, where='pre', lw=2, color='red')
	
	ax.set_zorder(10)
	ax.patch.set_visible(False)

	ax.set_xlabel('Data Points',fontsize=10)
	ax.set_ylabel('ISCF Probability',fontsize=10)
	# ax.set_xlim((-0.05,1.05))
	# ax.set_xticks(np.arange(0,1.05,0.2))
	ax.set_ylim((-0.05,1.05))
	ax.set_yticks(np.arange(0,1.05,0.5))
	ax.set_axisbelow(True)
	ax.grid(color='gray') #, linestyle='dashed')
	# ax.set_title(title_text,fontsize=10)
	# ax.legend()

	ax2.set_ylabel('Predicted MS', color='red')  # we already handled the x-label with ax1
	ax2.tick_params(axis='y', colors='red')
	ax2.set_ylim((-0.05,1.05))
	ax2.set_yticks(np.arange(0,1.05,1))
	ax2.set_yticklabels(['H','F'])

	fig_filename = '{}/sample_predictions__{}.png'.format(data_series_folder_path,data_series_id)
	fig.savefig(fig_filename, bbox_inches='tight')
	fig_filename = '{}/sample_predictions__{}.pdf'.format(data_series_folder_path,data_series_id)
	fig.savefig(fig_filename, bbox_inches='tight', dpi=200)
	plt.close('all')

	# sys.exit()

	sample_ids_list.append(sample_ids_data)
	sample_truths_list.append(truths_data)
	sample_preds_list.append(preds_data)
	sample_probs_list.append(probs_data)

	data_series_truth = truths_data[0]
	data_series_probs = np.mean(probs_data,axis=0)
	data_series_pred = np.argmax(data_series_probs)

	data_series_truths_list.append(data_series_truth)
	data_series_preds_list.append(data_series_pred)
	data_series_probs_list.append(data_series_probs)

	with open(data_series_predictions_file,'a') as f_data_series_predictions_file:
		f_data_series_predictions_file.write('{}\t'.format(data_series_id))
		f_data_series_predictions_file.write('{:d}\t'.format(data_series_truth))
		f_data_series_predictions_file.write('{:d}\t'.format(data_series_pred))
		f_data_series_predictions_file.write('{:.3f}\t'.format(data_series_probs[0]))
		f_data_series_predictions_file.write('{:.3f}\n'.format(data_series_probs[1]))

# sys.exit()
sample_ids_arr = np.concatenate(sample_ids_list, axis=0)
print('sample_ids_arr.shape:{}'.format(sample_ids_arr.shape))
sample_truths_arr = np.concatenate(sample_truths_list, axis=0)
print('sample_truths_arr.shape:{}'.format(sample_truths_arr.shape))
sample_preds_arr = np.concatenate(sample_preds_list, axis=0)
print('sample_preds_arr.shape:{}'.format(sample_preds_arr.shape))
sample_probs_arr = np.concatenate(sample_probs_list, axis=0)
print('sample_probs_arr.shape:{}'.format(sample_probs_arr.shape))
data_series_truths_arr = np.stack(data_series_truths_list, axis=0)
print('data_series_truths_arr.shape:{}'.format(data_series_truths_arr.shape))
data_series_preds_arr = np.stack(data_series_preds_list, axis=0)
print('data_series_preds_arr.shape:{}'.format(data_series_preds_arr.shape))
data_series_probs_arr = np.stack(data_series_probs_list, axis=0)
print('data_series_probs_arr.shape:{}'.format(data_series_probs_arr.shape))


# collect sample level statistics
temp_truth = sample_truths_arr
temp_pred = sample_preds_arr
temp_prob_pos = sample_probs_arr[:,1]

conf_mat = confusion_matrix(temp_truth, temp_pred, labels=[0,1])
conf_mat_filename = '{}/sample_level_cm.txt'.format(data_folder_path)
np.savetxt(conf_mat_filename, conf_mat, fmt='%d', delimiter='\t')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/sample_level_cm_normalized.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=False, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/sample_level_cm_unnormalized.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)
precision, recall, fscore, support = precision_recall_fscore_support(temp_truth, temp_pred, average='binary', labels=[0,1], pos_label=1)
# print(acc, precision, recall, fscore, support)

fpr, tpr, th = roc_curve(temp_truth, temp_prob_pos, pos_label=1)
auroc = auc(fpr, tpr)
# print(auroc)

distance_to_corner = fpr**2 + (1-tpr)**2
min_distance_index = np.argmin(distance_to_corner)
min_distance_th = th[min_distance_index]

print('min_distance_th:{}'.format(min_distance_th))

sorted_scores, auroc_lower, auroc_upper = BootStrap(temp_truth, temp_prob_pos, n_bootstraps=100)

title_text = 'AUROC={:.4f} ({:.4f} - {:.4f})'.format(auroc, auroc_lower, auroc_upper)

fig, ax = plt.subplots(figsize=(3,3))
ax.plot(fpr, tpr, lw=2, alpha=0.7, color='k')
# ax.plot(fpr, tpr, color='k', lw=2)
# ax.plot([0, 1], [0, 1], 'k--', lw=1)

ax.set_xlabel('False Positive Rate',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=10)
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
ax.set_axisbelow(True)
ax.grid(color='gray') #, linestyle='dashed')
ax.set_title(title_text,fontsize=10)
# ax.legend()

fig.tight_layout()
# plt.show()
fig.subplots_adjust(left=0.16, bottom=0.13, right=0.98, top=0.93, wspace=0.20 ,hspace=0.20 )
fig_filename = '{}/sample_level_roc.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)
fig_filename = '{}/sample_level_roc.pdf'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)
plt.close('all')

sample_level_statistics_filename = '{}/sample_level_statistics.txt'.format(data_folder_path)
with open(sample_level_statistics_filename, 'w') as f_sample_level_statistics_filename:
	f_sample_level_statistics_filename.write('# acc\tprecision\trecall\tfscore\tauroc\tauroc_lower\tauroc_upper\n')
	f_sample_level_statistics_filename.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(acc,precision,recall,fscore,auroc,auroc_lower,auroc_upper))




# collect data series level statistics - mpp
temp_truth = data_series_truths_arr
temp_pred = data_series_preds_arr
temp_prob_pos = data_series_probs_arr[:,1]

conf_mat = confusion_matrix(temp_truth, temp_pred, labels=[0,1])
conf_mat_filename = '{}/data_series_level_cm.txt'.format(data_folder_path)
np.savetxt(conf_mat_filename, conf_mat, fmt='%d', delimiter='\t')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/data_series_level_cm_normalized.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=False, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/data_series_level_cm_unnormalized.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')


acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)
precision, recall, fscore, support = precision_recall_fscore_support(temp_truth, temp_pred, average='binary', labels=[0,1], pos_label=1)

fpr, tpr, _ = roc_curve(temp_truth, temp_prob_pos, pos_label=1)
auroc = auc(fpr, tpr)
# print(auroc)

sorted_scores, auroc_lower, auroc_upper = BootStrap(temp_truth, temp_prob_pos, n_bootstraps=100)

title_text = 'AUROC={:.4f} ({:.4f} - {:.4f})'.format(auroc, auroc_lower, auroc_upper)

fig, ax = plt.subplots(figsize=(3,3))
ax.plot(fpr, tpr, lw=2, alpha=0.7, color='k')
# ax.plot(fpr, tpr, color='k', lw=2)
# ax.plot([0, 1], [0, 1], 'k--', lw=1)

ax.set_xlabel('False Positive Rate',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=10)
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
ax.set_axisbelow(True)
ax.grid(color='gray') #, linestyle='dashed')
ax.set_title(title_text,fontsize=10)
# ax.legend()

fig.tight_layout()
# plt.show()
fig.subplots_adjust(left=0.16, bottom=0.13, right=0.98, top=0.93, wspace=0.20 ,hspace=0.20 )
fig_filename = '{}/data_series_level_roc.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)
fig_filename = '{}/data_series_level_roc.pdf'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)
plt.close('all')

data_series_level_statistics_filename = '{}/data_series_level_statistics.txt'.format(data_folder_path)
with open(data_series_level_statistics_filename, 'w') as f_data_series_level_statistics_filename:
	f_data_series_level_statistics_filename.write('# acc\tprecision\trecall\tfscore\tauroc\tauroc_lower\tauroc_upper\n')
	f_data_series_level_statistics_filename.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(acc,precision,recall,fscore,auroc,auroc_lower,auroc_upper))



