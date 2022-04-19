# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

def check_split(line, sep, requried_field_num, check_mode, field_name=''):
	if not check_mode:
		return True
	assert sep, "must have separator." + (" field: %s." % field_name) if field_name else ""
	# if isinstance(sep, bytes):
	#   sep = bytes.decode(sep)
	# elif type(sep) != type(str):
	#   sep = str(sep).encode('utf-8')
	for one_line in line:
		field_num = len(one_line.split(sep))
		if field_name:
			assert_info = 'sep[%s] maybe invalid. field_num=%d, required_num=%d, field: %s, value: %s, ' \
			'please check separator and data.' % \
			(sep, field_num, requried_field_num, field_name, one_line)
		else:
			assert_info = 'sep[%s] maybe invalid. field_num=%d, required_num=%d, current line is: %s, ' \
			'please check separator and data.' % \
			(sep, field_num, requried_field_num, one_line)
		assert field_num == requried_field_num, assert_info
	return True

def check_string_to_number(field_vals, field_name, check_mode):
	if not check_mode:
		return True
	for val in field_vals:
		try:
			float(val)
		except:
			assert False, "StringToNumber ERROR: cannot convert string_to_number, field: %s, value: %s. " \
						  "please check data." % (field_name, val)
	return True

def check_size(field_vals1, field_vals2, field1, field2, check_mode):
	if not check_mode:
		return True
	assert len(field_vals1) == len(field_vals2), \
		"TagFeature Error: The size of %s not equal to the size of %s. Please check input: %s and %s." \
		% (field1, field2, field1, field2)
	return True

def check_train_step(pipeline_config):
	num_steps = pipeline_config.train_config.num_steps
	num_epochs = pipeline_config.data_config.num_epochs
	assert not (num_steps == 0 and num_epochs == 0), "num_steps and num_epochs cannot both be 0."

def check_sequence(pipeline_config_path, features):
	seq_att_groups = pipeline_config_path.model_config.seq_att_groups
	if not seq_att_groups:
		return
	for seq_att_group in seq_att_groups:
		seq_att_maps = seq_att_group.seq_att_map
		if not seq_att_maps:
			return
		for seq_att_map in seq_att_maps:
			assert len(seq_att_map.key) == len(seq_att_map.hist_seq), \
				'The size of hist_seq must equal to the size of key in one seq_att_map.'
			size_list = []
			for hist_seq in seq_att_map.hist_seq:
				cur_seq_size = len(features[hist_seq].values)
				size_list.append(cur_seq_size)
			hist_seqs = ' '.join(seq_att_map.hist_seq)
			assert len(set(size_list)) == 1, \
				'SequenceFeature Error: The size in [%s] should be consistent. Please check input: [%s].' % \
				(hist_seqs, hist_seqs)
