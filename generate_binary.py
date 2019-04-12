import numpy as np
import tensorflow as tf
import os
import copy
import pickle 
import matplotlib.pyplot as plt
import struct
import sys
import argparse
import importlib
import shutil
import errno
import time
import NeuroZERO
from scipy.optimize import linear_sum_assignment

param_filename = "_param.h"
skeleton_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skeleton/')
extended_MAIN_dir = os.path.join(skeleton_dir, 'extended_MAIN/')
extended_ACC_dir = os.path.join(skeleton_dir, 'extended_ACC/')
build_command = "eclipse -noSplash -data \"./\" -application com.ti.ccstudio.apps.projectBuild -ccs.configuration Debug -ccs.autoImport -ccs.projects "


def generate_param(network_name):
	p_f = open(network_name+param_filename, 'w')

	# layers
	f = open(NeuroZERO.layer_base_name + network_name, 'r')
	layer_num = struct.unpack('i', f.read(4))[0]
	p_f.write('#define MAX_NUM_OF_LAYER_DIM    10\n\n')
	p_f.write('uint16_t ' + network_name + '_layer[][MAX_NUM_OF_LAYER_DIM] = {\n')
	for l in range(layer_num):
		p_f.write('\t{ ')
		items = []
		while (1):
			item = struct.unpack('i', f.read(4))[0]
			items.append(item)
			if item == 0:
				break
		if len(items) >= 2 and items[1] != 0:
			items[1], items[0] = items[0], items[1]
		if len(items) >= 4:
			del items[-1]
			items.append(1)
			items.append(1)
			items.append(0)

		for i in range(len(items)):
			p_f.write(str(items[i]) + ', ')

		p_f.write('},\n')
	p_f.write('};\n\n')
	f.close()

	# weight
	p_f.write("#pragma PERSISTENT(" + network_name + "_weights)\n")
	p_f.write("_q " + network_name + "_weights[] = {\n\t")
	f = open(NeuroZERO.weight_base_name + network_name + '_q', 'r')
	weight_len = struct.unpack('i', f.read(4))[0]
	for i in range(0, weight_len):
		weight = struct.unpack('h', f.read(2))[0]
		p_f.write(str(weight))
		p_f.write(", ")
		if (i+1) % 20 == 0:
			p_f.write('\n\t')
	f.close()
	p_f.write("\n\t};\n")

	# bias
	p_f.write("\n#pragma PERSISTENT(" + network_name + "_biases)\n")
	p_f.write("_q " + network_name + "_biases[] = {\n\t")
	f = open(NeuroZERO.bias_base_name + network_name + '_q', 'r')
	bias_len = struct.unpack('i', f.read(4))[0]
	for i in range(0, bias_len):
		bias = struct.unpack('h', f.read(2))[0]
		p_f.write(str(bias))
		p_f.write(", ")
		if (i+1) % 20 == 0:
			p_f.write('\n\t')
	f.close()
	p_f.write("\n\t};\n")

	p_f.close()
	return network_name+param_filename

def copyAll(src, dst):
	try:
		shutil.copytree(src, dst)
	except OSError as exc: # python >2.5
		if exc.errno == errno.ENOTDIR:
			shutil.copy(src, dst)
		else:
			raise

def main(args):
	if args.mode == 'ext':
		shutil.rmtree('.metadata')
		# main MCU
		MAIN_src_dir = extended_MAIN_dir
		MAIN_target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.basename(os.path.normpath(MAIN_src_dir)))
		if not os.path.exists(MAIN_target_dir):
			copyAll(MAIN_src_dir, MAIN_target_dir)
			print MAIN_target_dir + ' created'
	
		filename = generate_param(NeuroZERO.baseline_network_name)
		if not os.path.exists(os.path.join(MAIN_target_dir, filename)):
			copyAll(filename, MAIN_target_dir)
			os.remove(filename)
			print filename + ' generated and copied'

		filename = generate_param(NeuroZERO.extended_network_name)
		if not os.path.exists(os.path.join(MAIN_target_dir, filename)):
			copyAll(filename, MAIN_target_dir)
			os.remove(filename)
			print filename + ' generated and copied'

		print 'start compiling main MCU'
		print build_command + os.path.basename(os.path.normpath(MAIN_target_dir))
		os.system(build_command + os.path.basename(os.path.normpath(MAIN_target_dir)))

		# accelerator 
		ACC_src_dir = extended_ACC_dir
		ACC_target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.basename(os.path.normpath(ACC_src_dir)))
		if not os.path.exists(ACC_target_dir):
			copyAll(ACC_src_dir, ACC_target_dir)
			print ACC_target_dir + ' created'
	
		filename = generate_param(NeuroZERO.extended_network_name)
		if not os.path.exists(os.path.join(ACC_target_dir, filename)):
			copyAll(filename, ACC_target_dir)
			os.remove(filename)
			print filename + ' generated and copied'

		print 'start compiling accelerator'
		print build_command + os.path.basename(os.path.normpath(ACC_target_dir))
		os.system(build_command + os.path.basename(os.path.normpath(ACC_target_dir)))


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', type=str,	help='mode', default='s')
	# network
	# s: show the current status of NeuroZERO (--mode=s)
	# c: construct a base network (--mode=c --data --layers)
	# d: destruct a network 
	# t: train a network
	# i: inference of a network
	# e: export weights and bias to a file

	parser.add_argument('--layers', type=str, help='layers', default=None)
	# CNN layer: 4 dimensions (height, width, depth, num of filters) [ex. 3,3,3,8]
	# Fully-connected layer: any dimension less than 4 [ex. 128 / 128*128 / 128*128*1]

	parser.add_argument('--data', type=str, help='data', default=None)
	parser.add_argument('--network_no', type=int, help='network_no', default=-1)
	parser.add_argument('--base_network_no', type=int, help='base_network_no', default=-1)
	parser.add_argument('--ensemble_network_no', type=str, help='ensemble_network_no', default=None)
	parser.add_argument('--vnn_no', type=int, help='vnn_no', default=-1)
	parser.add_argument('--name', type=str, help='name', default=None)
	parser.add_argument('--save', type=bool, help='save NeuroZERO?', default=True)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
