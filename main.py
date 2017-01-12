"""
LSTM-MLB 1.0.

Usage:
	main.py bs_id <bs_id> [--length=<sequence_length>] [--offset=<offset>] [--lr=<learning_rate>] [--bs=<batch_size>] [--epoch=<epoch>] [--ns=<num_step>] [--dm=<dataset_mode>]
	main.py bs_id <bs_id> data_mode <data_mode> data_length <data_length> data_offset <data_offset>
	main.py random <random> [--length=<sequence_length>] [--offset=<sequence_length>] [--lr=<learning_rate>] [--bs=<batch_size>] [--epoch=<epoch> ]

Options:
	-h --help
"""
# for test use command below:
# -------------
# Validate lstm in MLB model

# dataset parameter:

# 	bs-id
#	sequence-lengthd
#	offset

# tran parameter:
#	learning rate
#	batch size
#	epoch

from __future__ import print_function
import dataset
import lstm
import lstm_simple_rnn_v1
import config
from docopt import docopt

def batch_process_data(x, y):
	for i in range (x, y+1):
		print("Processing data")
		arguments["<bs_id>"] = i
		config.read(arguments)
		config.test()
		dataset.process_data(config.bs_id, config.data_length, config.data_offset, config.data_mode)

if __name__ == '__main__':
	arguments = docopt(__doc__)
	print(arguments)
	#For train
	config.read(arguments)
	if (config.data_mode > 0):
		config.test()
		batch_process_data(config.bs_id, config.bs_id)
	else:
		lstm.prepare_dataset(config.dataset_file, config.neighbor_set, config.sequence_length, config.offset)
		lstm.construct_and_train(config.epoch, 1)

		#lstm_simple_rnn_v1.prepare_dataset(config.dataset_file, config.)
	