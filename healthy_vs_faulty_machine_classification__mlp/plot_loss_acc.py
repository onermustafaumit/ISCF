import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot the loss vs iteration and accuracy vs iteration for given data file')
parser.add_argument('--data_file', help='Data file path', dest='data_file')
parser.add_argument('--step_size', default=1, type=int, help='Data file path', dest='step_size')
parser.add_argument('--filter_size', default=1, type=int, help='Data file path', dest='filter_size')
FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')

# steps = data_arr[:,0]
steps = np.arange(data_arr.shape[0]) + 1
train_acc = data_arr[:,1]
train_loss = data_arr[:,2]
val_acc = data_arr[:,3]
val_loss = data_arr[:,4]
test_acc = data_arr[:,5]
test_loss = data_arr[:,6]


def moving_avg_filter(data_arr, w):
	data_arr_cumsum = np.cumsum(data_arr)
	data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
	data_arr_filtered = data_arr_cumsum[w-1:]/w

	return data_arr_filtered

if w>1:
	steps = steps[w-1:]
	train_acc = moving_avg_filter(train_acc,w)
	train_loss = moving_avg_filter(train_loss,w)
	val_acc = moving_avg_filter(val_acc,w)
	val_loss = moving_avg_filter(val_loss,w)
	test_acc = moving_avg_filter(test_acc,w)
	test_loss = moving_avg_filter(test_loss,w)


ind_start = 0
ind_step = FLAGS.step_size
if ind_step>1:
	ind_start = ind_step - 1
ind_end = len(steps) + 1

fig,ax = plt.subplots(1,2,figsize=(8,3))
ax[0].plot(steps[ind_start:ind_end:ind_step], train_loss[ind_start:ind_end:ind_step], 'r', label="train")
ax[0].plot(steps[ind_start:ind_end:ind_step], val_loss[ind_start:ind_end:ind_step], 'b', label="val")
ax[0].plot(steps[ind_start:ind_end:ind_step], test_loss[ind_start:ind_end:ind_step], 'cyan', label="test")
# ax[0].set_title('loss vs epoch')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].grid(linestyle='--')
ax[0].legend()


ax[1].plot(steps[ind_start:ind_end:ind_step], train_acc[ind_start:ind_end:ind_step], 'r', label="train")
ax[1].plot(steps[ind_start:ind_end:ind_step], val_acc[ind_start:ind_end:ind_step], 'b', label="val")
ax[1].plot(steps[ind_start:ind_end:ind_step], test_acc[ind_start:ind_end:ind_step], 'cyan', label="test")
# ax[1].set_title('acc vs epoch')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].grid(linestyle='--')
ax[1].legend()

fig.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.98, wspace=0.24 ,hspace=0.20 )
fig.savefig('{}__loss_acc.png'.format(FLAGS.data_file[:-4]), transparent=True)

plt.show()




	