

from __future__ import print_function

# BS and UE parameter
BS_SUM = 1000
UE_SUM = 20000
EPOCH_SUM = 2000

# Data parameter

dataset_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
bs_id = []
sample_count = 0
neighbor_count = 25
neighbor_set = []
dataset_file = ""
dataset_mode = 1

# Training parameter
epoch = 10
batch_size = 1
learning_rate = 0.001
batch_count = 1

# LSTM model papameter
sequence_length = 10
num_step = 5
offset = 2
layer_size = 3
state_size = 100
keep_prob = 0.8

# To do:Wheter randomly choose BS to train
random = 0

process_data = 0
display_step = 10

# Process data
data_mode = 0
data_length = 0
data_offset = 0


import dataset

def read(arguments):
	global dataset_id_list
	global bs_id
	global epoch
	global batch_size
	global learning_rate
	global random
	global sequence_length
	global offset
	global dataset_file
	global dataset_mode
	global total_sample
	global batch_count
	global num_step
	global process_data
	global data_mode
	global data_length
	global data_offset

	bs_id = int(arguments["<bs_id>"])
	if (arguments["--bs"]):
		batch_size = int(arguments["--bs"])
	if (arguments["--lr"]):
		learning_rate = float(arguments["--lr"])
	if (arguments["--length"]):
		sequence_length = int(arguments["--length"])
	if (arguments["--offset"]):
		offset = int(arguments["--offset"])
	if (arguments["--epoch"]):
		epoch = int(arguments["--epoch"])
	if (arguments["--ns"]):
		num_step = int(arguments["--ns"])
	if (arguments["--dm"]):
		dataset_mode = int(arguments["--dm"])
	if(arguments["data_mode"]):
		data_mode = int(arguments["<data_mode>"])
		data_length = int(arguments["<data_length>"])
		data_offset = int(arguments["<data_offset>"])
		dataset_file = "data/lstm_" + str(data_mode) + "/" + str(bs_id) + ".lstm"
	else:
		dataset_file = "data/lstm_" + str(dataset_mode) + "/" + str(bs_id) + ".lstm"
		data_mode = 0
	
	if(not data_mode):
		dataset.calculate_neighbor(bs_id, dataset_mode)
		neighbor_set.sort()
	batch_count = sample_count / batch_size

def test():
	print("Parameters:")
	print("bs_id = ", bs_id)
	print("dataset_file = ", dataset_file)
	if(not data_mode):
		print("epoch = ", epoch)
		print("batch_size = ", batch_size)
		print("batch_count = ", batch_count)
		print("lerning rate = ", learning_rate)
		print("random = ", random)
		print("sequence_length = ", sequence_length)
		print("offset = ", offset)
		print("neighbor_count = ", neighbor_count)
		print("neighbor_set = ", neighbor_set)
		print("sample_count = ", sample_count)
		print("state_size = ", state_size)
		print("num_step = ", num_step)
		print("dataset_mode = ", dataset_mode)
	if (data_mode):
		print("data_mode = ", data_mode)
		print("data_length = ", data_length)
		print("data_offset = ", data_offset)

