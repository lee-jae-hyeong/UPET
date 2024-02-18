"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from sklearn.utils import shuffle

import logging
import numpy as np
import os
import random
import torch
from sklearn.metrics import accuracy_score
random_seed = 42
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)


logger = logging.getLogger('UST_RES')

def get_BALD_acquisition(y_T):

	expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + 1e-10), axis=-1), axis=0) 
	expected_p = np.mean(y_T, axis=0)
	entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
	BALD_acq = (entropy_expected_p - expected_entropy)
	# BALD_acq = np.where(BALD_acq < 0 , 1, BALD_acq)
	# BALD_acq = np.log(BALD_acq)**2
	# return (entropy_expected_p - expected_entropy)
	return BALD_acq

def sample_by_bald_difficulty(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
	p_norm = p_norm / np.sum(p_norm)
	indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
	X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices], "attention_mask": X["attention_mask"][indices]}
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_s, w_s


def sample_by_bald_easiness(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by easy BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), (1. - BALD_acq)/np.sum(1. - BALD_acq))
	p_norm = p_norm / np.sum(p_norm)
	logger.info (p_norm[:10])
	indices = np.random.choice(len(X['input_ids']), num_samples, p=p_norm, replace=False)
	X_s = {"input_ids": X["input_ids"][indices], "token_type_ids": X["token_type_ids"][indices], "attention_mask": X["attention_mask"][indices]}
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_s, w_s


def sample_by_bald_class_easiness(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T, alpha, cb_loss=True, true_label = None, active_learning= False, active_number = 16):

	assert (alpha >= 0) & (alpha <= 1), "alpha should be between 0 and 1"

	logger.info ("Sampling by easy BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	#BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)
	# 2024.01.19 reliable examples sampling 코드 구현 미비로 인해 추가
	# add by ljh
	scf_index = np.argmax(y_mean, axis = 1)
	scf = y_mean[np.arange(len(y_mean)), scf_index]
	print(scf)
	sct = (BALD_acq)
	print(sct)

	
	array_BALD_acq = np.array(BALD_acq)
	np.savetxt('/content/array.txt', array_BALD_acq)
	res_score = ((alpha * scf) + ((1-alpha) * sct)) / (np.sum(alpha * scf) + np.sum((1 - alpha) * sct))
	
	BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)
	logger.info (BALD_acq)
	logger.info (f'res_score: {res_score}')

	samples_per_class = num_samples // num_classes
	
	# for label in range(num_classes):
 #        	total_var += np.mean(y_var[y==label])
		
	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, X_s_mask_pos, y_s, w_s = [], [], [], [], [], []
	not_sample = 0
	total_var = 0

	total_res_score = np.sum(res_score)

		
	for label in range(num_classes):
		# X_input_ids, X_token_type_ids, X_attention_mask = np.array(X['input_ids'])[y == label], np.array(X['token_type_ids'])[y == label], np.array(X['attention_mask'])[y == label]
		X_input_ids, X_attention_mask = np.array(X['input_ids'])[y == label], np.array(X['attention_mask'])[y == label]
		if "token_type_ids" in X.features:
			X_token_type_ids = np.array(X['token_type_ids'])[y == label]
		if "mask_pos" in X.features:
			X_mask_pos = np.array(X['mask_pos'])[y == label]
		y_ = y[y==label]
		y_var_ = y_var[y == label]
		print('분산 평균 : ', np.mean(y_var_))
		# regular_var = 1 - (np.mean(y_var_) / total_var)
		# print('분산 정규화 : ', regular_var, '샘플링 수 :', round(num_samples*regular_var), num_samples*regular_var)

		# p = y_mean[y == label]
		#2024.01.19 주석 처리
		#p_norm = BALD_acq[y==label]
		p_norm = res_score[y==label]

		ratio = np.sum(p_norm)/total_res_score
		#print('res_count : ', np.sum(p_norm)/total_res_score, num_samples * ratio)
		
		print("res 평균 : ", np.mean(p_norm))

		if active_learning:
			sorted_indices = np.argsort(p_norm)
			indices = sorted_indices[:active_number]

		else :
			p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
			p_norm = p_norm/np.sum(p_norm)
			
			if len(X_input_ids) < samples_per_class:
				logger.info ("Sampling with replacement.")
				replace = True
			else:
				replace = False
			# print("====== label: {} ======".format(label))
			# print("len(X_input_ids)=", len(X_input_ids))
			# print("samples_per_class=", samples_per_class)
			# print("p_norm=", p_norm)
			# print("replace=", replace)
			if len(X_input_ids) == 0: # add by wjn
				not_sample += 1
				continue
			indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
	
			if not true_label is None:
				true_label_ = true_label[y==label]
				print('정확도 : ', accuracy_score(true_label_[indices], y_[indices]))
				print('실제 : ', true_label_[indices])
				print('수도 레이블 : ', y_[indices])
			# add by ljh
			if len(set(indices)) != samples_per_class:
				print("samples_per_class : {}".format(samples_per_class))
				print("sampling_count : {}".format(len(set(indices))))
				
				logger.info ("{}_Not Enough data ratio".format(len(set(indices)), samples_per_class)) 
				print("{}_Not Enough data ratio".format(len(set(indices))/samples_per_class))
				if cb_loss:
					# cb_loss 적용 시, 중복 제거
					indices = np.array(list(set(indices)))
				
			
		X_s_input_ids.extend(X_input_ids[indices])
		# X_s_token_type_ids.extend(X_token_type_ids[indices])
		X_s_attention_mask.extend(X_attention_mask[indices])
		if "token_type_ids" in X.features:
			X_s_token_type_ids.extend(X_token_type_ids[indices])
		if "mask_pos" in X.features:
			X_s_mask_pos.extend(X_mask_pos[indices])
		y_s.extend(y_[indices])
		w_s.extend(y_var_[indices][:,0])

	print("{}_SAMPLING_FAIL_COUNT".format(not_sample))
	# X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
	if "token_type_ids" in X.features and "mask_pos" not in X.features:
		X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
	elif "token_type_ids" not in X.features and "mask_pos" in X.features:
		X_s_input_ids, X_s_mask_pos, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_mask_pos, X_s_attention_mask, y_s, w_s)
	elif "token_type_ids" in X.features and "mask_pos" in X.features:
		X_s_input_ids, X_s_token_type_ids, X_s_mask_pos, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_mask_pos, X_s_attention_mask, y_s, w_s)
	else:
		X_s_input_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_attention_mask, y_s, w_s)
	
	# return {'input_ids': np.array(X_s_input_ids), 'token_type_ids': np.array(X_s_token_type_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)
	
	pseudo_labeled_input = {
		'input_ids': np.array(X_s_input_ids), 
		'attention_mask': np.array(X_s_attention_mask)
	}
	if "token_type_ids" in X.features:
		pseudo_labeled_input['token_type_ids'] = np.array(X_s_token_type_ids)
	if "mask_pos" in X.features:
		pseudo_labeled_input['mask_pos'] = np.array(X_s_mask_pos)
	return pseudo_labeled_input, np.array(y_s), np.array(w_s)


def sample_by_bald_class_difficulty(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	samples_per_class = num_samples // num_classes
	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = [], [], [], [], []
	for label in range(num_classes):
		X_input_ids, X_token_type_ids, X_attention_mask = X['input_ids'][y == label], X['token_type_ids'][y == label], X['attention_mask'][y == label]
		y_ = y[y==label]
		y_var_ = y_var[y == label]		
		p_norm = BALD_acq[y==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		if len(X_input_ids) < samples_per_class:
			replace = True
			logger.info ("Sampling with replacement.")
		else:
			replace = False
		indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
		X_s_input_ids.extend(X_input_ids[indices])
		X_s_token_type_ids.extend(X_token_type_ids[indices])
		X_s_attention_mask.extend(X_attention_mask[indices])
		y_s.extend(y_[indices])
		w_s.extend(y_var_[indices][:,0])
	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
	return {'input_ids': np.array(X_s_input_ids), 'token_type_ids': np.array(X_s_token_type_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)
