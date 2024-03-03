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



logger = logging.getLogger('UST_RES')

def margin_sampling(y_mean):
    """
    Margin Sampling을 사용하여 다음으로 라벨링할 샘플을 선택합니다.
    
    매개변수:
        probabilities (2D 배열): 각 샘플에 대한 클래스별 예측 확률을 포함하는 2D 배열입니다.
                                  각 행은 하나의 샘플을 나타내며, 각 열은 클래스를 나타냅니다.
    
    반환값:
        selected_index (int): 선택된 샘플의 인덱스입니다.
    """
    # 각 샘플의 예측된 클래스 인덱스
    predicted_classes = np.argmax(y_mean, axis=1)
    
    # 각 샘플의 최대 예측 확률
    max_probabilities = np.max(y_mean, axis=1)
    
    # 각 샘플의 두 번째로 큰 예측 확률
    second_max_probabilities = np.partition(y_mean, -2, axis=1)[:, -2]
    
    # Margin을 계산
    margins = max_probabilities - second_max_probabilities

    return margins
	
def get_BALD_acquisition(y_T, up_scale = False):

	expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + 1e-10), axis=-1), axis=0) 
	expected_p = np.mean(y_T, axis=0)
	entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
	BALD_acq = (entropy_expected_p - expected_entropy)

	# BALD SCORE의 원본 버전
	if not up_scale:
		print('발드 그대로 출력')
		return BALD_acq
	# BALD SCORE의 새로운 버전
	# BALD SCORE의 경우, 차이가 매우 작기 때문에 정규화를 거치고 나면 상대적으로 BALD의 차이가 가지는 의미가 퇴색됨. 그러므로, BALD Score를 확실하게 하기 위하여 up_scale을 진행.
	# np.log(bald) ** 2
	else:
		print('발드 업스케일링 출력')
		BALD_acq = np.where(BALD_acq < 0 , 1, BALD_acq)
		BALD_acq = -np.log(BALD_acq + 1e-10)

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


def sample_by_bald_class_easiness(tokenizer, X, y_mean, y_var, y, num_samples, num_classes, y_T, alpha, cb_loss=True, true_label = None, active_learning= False, active_number = 16, uncert = True, up_scale= True, c_type='BALD'):

	assert (alpha >= 0) & (alpha <= 1), "alpha should be between 0 and 1"

	logger.info ("Sampling by easy BALD acquisition function per class")
	
	if up_scale:
		print('up_scale = True 따른 BALD 업스케일링 진행')
		BALD_acq = get_BALD_acquisition(y_T, up_scale = True)
		sct = (BALD_acq)
	else:
		print('up_scale = False 따른 BALD 업스케일링 진행하지 않음.')
		BALD_acq = get_BALD_acquisition(y_T, up_scale = False)
		sct = (1. - BALD_acq)
		
	# 2024.01.19 reliable examples sampling 코드 구현 미비로 인해 추가
	# add by ljh
	# y_mean의 가장 확률적으로 높은 값이 레이블로 할당되기 때문에, confidence 측정을 위해 추출
	
	scf_index = np.argmax(y_mean, axis = 1)
	scf = y_mean[np.arange(len(y_mean)), scf_index]

	#res_score 정규화
	#res_score : confidence와 certainty를 조합하여 신뢰도에 대한 점수(alpha를 통해 조절)
	res_score = ((alpha * scf) + ((1-alpha) * sct)) / (np.sum(alpha * scf) + np.sum((1 - alpha) * sct))
	
	logger.info (BALD_acq)
	logger.info (f'res_score: {res_score}')

	samples_per_class = num_samples // num_classes

	active_X_s_input_ids, active_X_s_token_type_ids, active_X_s_attention_mask, active_X_s_mask_pos, active_y_s, active_w_s, active_X_idxs = [], [], [], [], [], [], []
	
	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, X_s_mask_pos, y_s, w_s = [], [], [], [], [], []
	not_sample = 0

	if c_type == "entropy":
		print("엔트로피 기반의 샘플링 전략")
		entropy_score = - np.sum(y_mean * np.log(y_mean + 1e-10), axis=-1)
		res_score = 1 - (entropy_score / np.sum(entropy_score))
		
	elif c_type == "var":
		print("예측 분산 기반의 샘플링 전략")
		y_var_mean = np.mean(y_var, axis=1)
		res_score = 1 - ((y_var_mean) / np.sum(y_var_mean))
		
	elif c_type == "marginal":
		print("Marginal 기반의 샘플링 전략")
		margins = margin_sampling(y_mean)
		res_score = margins / np.sum(margins)
		
	if active_learning:
		for label in range(num_classes):
			
			p_norm = res_score[y==label]
			p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
			p_norm = p_norm/np.sum(p_norm)
			
			X_input_ids, X_attention_mask = np.array(X['input_ids'])[y == label], np.array(X['attention_mask'])[y == label]
			X_idx = np.array(X['idx'])[y == label]
			
			if "token_type_ids" in X.features:
				X_token_type_ids = np.array(X['token_type_ids'])[y == label]
			if "mask_pos" in X.features:
				X_mask_pos = np.array(X['mask_pos'])[y == label]
				
			y_ = y[y==label]
			y_var_ = y_var[y == label]
			true_label_ = true_label[y==label]
			
			print('ACTIVE_LEARNING_START')
			if uncert:
				print('ACTIVE_LEARNING_UNCERTAINTY')
				sorted_indices = np.argsort(p_norm)
				indices = sorted_indices[:active_number]
			else:
				print('ACTIVE_LEARNING_RANDOM_SAMPLE')
				indices = np.random.choice(len(p_norm), active_number, replace=False)
				
			print(len(indices))
			# print(X)
			# print(X['idx'])
			y_[indices] = true_label_[indices]
			# active_X_idxs.extend(np.array(X['idx'])[indices])
			active_X_s_input_ids.extend(X_input_ids[indices])
			active_X_s_attention_mask.extend(np.array(X_attention_mask)[indices])
			active_X_idxs.extend(X_idx[indices])
			
		
			if "token_type_ids" in X.features:
				active_X_s_token_type_ids.extend(X_token_type_ids[indices])
			if "mask_pos" in X.features:
				active_X_s_mask_pos.extend(X_mask_pos[indices])
				
			active_y_s.extend(y_[indices])
			#tmp_y_var = np.zero(len(active_number))
			#active_w_s.extend(tmp_y_var)
			
			#indices_to_keep = np.logical_not(np.isin(np.arange(len(res_score)), indices))
	
			check_number = len(y_)
			print('ACTIVE_SAMPLING 데이터 숫자 : ', check_number)
			#X = X[indices_to_keep]
			#y_var = y_var[indices_to_keep]
			#y = y[indices_to_keep]
			
			#print('ACTIVE_SAMPLING 이후의 데이터 숫자 : ', len(X['idx']), len(y_var), len(y))
			#print('숫자 일치 유무 : ', check_number - active_number == len(y))
			
		active_labeled_input = {
			'input_ids': np.array(active_X_s_input_ids), 
			'attention_mask': np.array(active_X_s_attention_mask)
		}
		if "token_type_ids" in X.features:
			active_labeled_input['token_type_ids'] = np.array(active_X_s_token_type_ids)
		if "mask_pos" in X.features:
			active_labeled_input['mask_pos'] = np.array(active_X_s_mask_pos)
				
		
	for label in range(num_classes):
		# X_input_ids, X_token_type_ids, X_attention_mask = np.array(X['input_ids'])[y == label], np.array(X['token_type_ids'])[y == label], np.array(X['attention_mask'])[y == label]
		X_input_ids, X_attention_mask = np.array(X['input_ids'])[y == label], np.array(X['attention_mask'])[y == label]
		#X_idx = np.array(X['idx'])[y == label]
		if "token_type_ids" in X.features:
			X_token_type_ids = np.array(X['token_type_ids'])[y == label]
		if "mask_pos" in X.features:
			X_mask_pos = np.array(X['mask_pos'])[y == label]
			
		y_ = y[y==label]
		y_var_ = y_var[y == label]
		print('분산 평균 : ', np.mean(y_var_))

		# p = y_mean[y == label]
		#2024.01.19 주석 처리
		#p_norm = BALD_acq[y==label]
		p_norm = res_score[y==label]


		print("res 평균 : ", np.mean(p_norm))

		# if active_learning:
		# 	if uncert:
		# 		print('ACRIVE_LEARNING_UNCERTAINTY_BASED')
		# 		sorted_indices = np.argsort(p_norm)
		# 		indices = sorted_indices[:active_number]
		# 		true_label_ = true_label[y==label]
		# 		y_[indices] = true_label_[indices]
		# 		X_idxs.extend(X_idx[indices])

		# 	else:
		# 		print('ACRIVE_LEARNING_RANDOM_BASED')
		# 		indices = np.random.choice(len(X_input_ids), active_number, replace=False)
		# 		true_label_ = true_label[y==label]
		# 		y_[indices] = true_label_[indices]
		# 		X_idxs.extend(X_idx[indices])


		# else :
		print('SELF_TRAINING')

		if uncert:
			p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
			p_norm = p_norm/np.sum(p_norm)

			# true_label_ = true_label[y==label]
			# y_[indices] = true_label_[indices]
			if len(X_input_ids) == 0: # add by wjn
				not_sample += 1
				continue

			# UST, UPET는 Active Learning이 없기 때문에, 다양성을 위해 확률적 샘플링을 진행하지만, UAST는 ST이전에 AL을 진행하기 때문에 다양성을 확보할 수 있음.
			# self_training sample_selection 1번째 컨디션 : 모수의 갯수가 샘플링하는 갯수의 2배 이하인 경우엔, 모수가 충분치 않다고 판단 / 확률적 랜덤 샘플링 진행 
			if len(X_input_ids) < (samples_per_class * 2):
				logger.info ("Sampling with replacement.")
				replace = True
				indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
				
			# self_training sample_selection 2번째 컨디션 : 모수의 갯수가 샘플링하는 갯수의 2배 이상인 경우엔, 모수가 충분하다고 판단 / 상위 N개를 추출
			else:
				replace = False
				indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
				# sorted_indices = np.argsort(-p_norm)
				# indices = sorted_indices[:samples_per_class]

			if not true_label is None:
				true_label_ = true_label[y==label]
				print(label,' 의 정확도 : ',accuracy_score(true_label_[indices], y_[indices]))
				# print('실제 : ', true_label_[indices])
				# print('수도 레이블 : ', y_[indices])
			
			# add by ljh
			# 샘플링 했을 때, 얼마나 부족한지 체크
			if len(set(indices)) != samples_per_class:
				indices = np.array(list(set(indices)))
				print("samples_per_class : {}".format(samples_per_class))
				print("sampling_count : {}".format(len(set(indices))))
				
				logger.info ("{}_Not Enough data ratio".format(len(set(indices)), samples_per_class)) 
				print("{}_Not Enough data ratio".format(len(set(indices))/samples_per_class))

		else:
			print('NOT_UNCERTAINTY_SAMPLING AND RANDOM_SAMPLING BY SELF_TRAINING')
			
			if len(X_input_ids) < (samples_per_class * 2):
				logger.info ("Sampling with replacement.")
				replace = True
			else:
				replace = False

			indices = np.random.choice(len(X_input_ids), samples_per_class, replace=replace)
			
					
			
			# if len(X_input_ids) < samples_per_class:
			# 	logger.info ("Sampling with replacement.")
			# 	replace = True
			# else:
			# 	replace = False
			# print("====== label: {} ======".format(label))
			# print("len(X_input_ids)=", len(X_input_ids))
			# print("samples_per_class=", samples_per_class)
			# print("p_norm=", p_norm)
			# print("replace=", replace)
			# if len(X_input_ids) == 0: # add by wjn
			# 	not_sample += 1
			# 	continue
			# indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)

			# 실제 노이즈가 얼마나 이루어지는지 체크하기 위함. 학습에 관여 x
			if not true_label is None:
				true_label_ = true_label[y==label]
				print(label,' 의 정확도 : ',accuracy_score(true_label_[indices], y_[indices]))
				# print('실제 : ', true_label_[indices])
				# print('수도 레이블 : ', y_[indices])
			
			# add by ljh
			# 샘플링 했을 때, 얼마나 부족한지 체크
			if len(set(indices)) != samples_per_class:
				indices = np.array(list(set(indices)))
				print("samples_per_class : {}".format(samples_per_class))
				print("sampling_count : {}".format(len(set(indices))))
				
				logger.info ("{}_Not Enough data ratio".format(len(set(indices)), samples_per_class)) 
				print("{}_Not Enough data ratio".format(len(set(indices))/samples_per_class))
				# if cb_loss:
				# 	# cb_loss 적용 시, 중복 제거
			# indices = np.array(list(set(indices)))
				
			
		X_s_input_ids.extend(X_input_ids[indices])
		# X_s_token_type_ids.extend(X_token_type_ids[indices])
		X_s_attention_mask.extend(X_attention_mask[indices])
		if "token_type_ids" in X.features:
			X_s_token_type_ids.extend(X_token_type_ids[indices])
		if "mask_pos" in X.features:
			X_s_mask_pos.extend(X_mask_pos[indices])
		y_s.extend(y_[indices])
		w_s.extend(y_var_[indices][:,label])
		#X_idxs.extend(np.ones(len(indices))* -1)

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
		
	return pseudo_labeled_input, np.array(y_s), np.array(w_s), active_labeled_input, np.array(active_y_s), np.array(active_w_s), active_X_idxs


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
