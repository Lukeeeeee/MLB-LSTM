# Return dataset for lstm to train and test

# parameter:
# 	bs-id
#	sequence-length
#	offset
from __future__ import print_function
import config

bs_map = [[0 for i in range(config.BS_SUM)] for j in range(config.BS_SUM)]
ue_bs_list = [[-1] for j in range(config.UE_SUM)]
bs_ue_path = [[] for j in range(config.BS_SUM)]

def clear():
	global bs_map
	global ue_bs_list
	global bs_ue_path
	bs_map = [[0 for i in range(config.BS_SUM)] for j in range(config.BS_SUM)]
	ue_bs_list = [[-1] for j in range(config.UE_SUM)]
	bs_ue_path = [[] for j in range(config.BS_SUM)]

def get_ue_path(sequence_length, bs_id, offset, mode):
	print("Get UE path")
	for i in range(config.UE_SUM):
		bs_list = ue_bs_list[i]
		if (len(bs_list) > sequence_length + offset):
			list_length = len(bs_list)
			for start_index, bs_id in enumerate(bs_list):
				if (start_index + sequence_length + offset -1 < list_length):
					end_bs = bs_list[start_index + sequence_length -1]
					if(end_bs != bs_id):
						continue
					else:
						append_list = bs_list[start_index:start_index + sequence_length + offset]
						for index, id in enumerate(append_list):
							if (bs_map[end_bs][id] != 1):
								append_list[index] = 0
						bs_ue_path[end_bs].append(append_list)
def out_sample(file_id, bs_id, file, mode):
	print("Out write data")
	bs_id = int(bs_id)
	sum = len(bs_ue_path[bs_id])
	if(file_id <= 1):
		file_out = open(file, "w")
	else:
		file_out = open(file, "a")
	print (sum, file = file_out)
	for list in bs_ue_path[bs_id]:
		for i in range(len(list) - 1):
			print(list[i], file = file_out, end = " ")
		print (list[len(list) -1], file = file_out)
	file_out.close()

def read(dataset, mode):
	print("Reading")
	file_in = open(dataset, "r")
	data = file_in.readlines()
	for line in data:
		item = line.split(" ")
		if (item[0] == "epoch"):
			epoch = int(item[1])
		elif (item[0] == "ue"):																																																																																																																																																																																																																																																																																																																																																																																																																																								
			current_ue = int(item[1])
		else:
			for bs in item:
				bs_id = int(bs)
				pre_bs_id = ue_bs_list[current_ue][len(ue_bs_list[current_ue]) - 1]
				if((mode == 2 and bs_id != pre_bs_id) or mode != 2):
					ue_bs_list[current_ue].append(bs_id)
					bs_map[bs_id][bs_id] = 1
					if(pre_bs_id != -1 and bs_id != -1):
						bs_map[pre_bs_id][bs_id] = 1;
						bs_map[bs_id][pre_bs_id] = 1;
	file_in.close()

def calculate_neighbor(center_bs_id, mode):
	print("Calculate neighbor")
	file_in = open(config.dataset_file, "r")
	data = file_in.readlines()
	neighbor_count = 0
	sample_count = 0
	neighbor_set = []
	clear()
	for index, line in enumerate(data):
		item = line.split(" ")
		if ((index == 0) or len(item) == 1):
			sample_count = int(item[0]) + sample_count
		else:
			for bs in item:
				bs_id = int(bs)
				if (bs_map[bs_id][bs_id] == 0):
					neighbor_count = neighbor_count + 1
					bs_map[bs_id][bs_id] = 1
					neighbor_set.append(bs_id)
	print("The total neighbor of bs id ", str(center_bs_id), "is ", str(neighbor_count))
	file_in.close()
	config.neighbor_count = neighbor_count
	config.neighbor_set = neighbor_set
	config.sample_count = sample_count
	#return neighbor_count, neighbor_set, sample_count

def process_data(bs_id, length, offset, mode):
	for id, file in enumerate(config.dataset_id_list):
		clear()
		file_in = "data/ue/" + str(file) + ".ue"
		file_out = config.dataset_file
		print("Process", file_in)
		read(file_in, mode)
		get_ue_path(length, bs_id, offset, mode)
		out_sample(id, bs_id, file_out, mode)
		calculate_neighbor(bs_id, mode)