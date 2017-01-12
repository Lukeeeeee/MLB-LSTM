"""
LSTM-MLB 1.0

Usage:
	data_pre_process.py
	data_pre_process.py dataset <dataset>...

Options:
	-h --help
"""


from __future__ import print_function

from docopt import docopt
import numpy as np
import re
import config

BS_SUM = config.BS_SUM
UE_SUM = config.UE_SUM
EPOCH_SUM = config.EPOCH_SUM


ue_id = [-1 for i in range(UE_SUM)]
ue_bs_list = [[-1 for i in range (EPOCH_SUM)] for j in range (UE_SUM)]

def add(epoch, bs, ue):
	if (ue != -1 and bs != -1):
		ue_id[ue] = 1;
		print (epoch, ue, bs)
		ue_bs_list[ue][epoch] = bs

def clear():
	global ue_id
	global ue_bs_list
	ue_id = [-1 for i in range(UE_SUM)]
	ue_bs_list = [[-1 for i in range (EPOCH_SUM)] for j in range (UE_SUM)]

def out_write(file_out, epoch):
	print ("epoch", epoch, file = file_out)
	for ue in range (UE_SUM):
		if (ue_id[ue] == 1):
			print("ue", ue, file = file_out)
			for i in range (epoch-1):
				print (ue_bs_list[ue][i], file = file_out, end = " ")
			print (ue_bs_list[ue][epoch-1], file = file_out)
		file_out.flush()

def read(arguments):
	for data in arguments['<dataset>']:
		data_in = "data/origin/" + data + ".origin"
		data_out = "data/ue/" + data + ".ue"
		file_in = open(data_in, "r")
		file_out = open(data_out, "w")
		clear()
		data = file_in.readlines()
		for line in data:
			item = line.split(" ")
			if (item[0] == "epoch:"):
				current_epoch = int(item[1])
				print ("epoch = ", current_epoch)
			elif (item[0] == "bs"):
				current_bs = int(item[2])
			elif (item[0] == "served"):
				current_bs_served_ue_sum = int(item[3])
			else:
				for id in item:
					if (not re.search("[^0-9]", id)):
						current_ue = int(id)
						add(current_epoch, current_bs, current_ue)
		total_epoch = current_epoch + 1;
		


		out_write(file_out, total_epoch)
		file_in.close()
		file_out.close()

if __name__ == '__main__':
	arguments = docopt(__doc__)
	print (arguments)
	read(arguments)
	



