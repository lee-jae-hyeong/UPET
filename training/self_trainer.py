import os
from typing import Dict, OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Union, Optional, Callable, List, Tuple
from transformers import Trainer
import datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import numpy as np
from typing import Optional, List
from datasets import Dataset, DatasetInfo, NamedSplit, DatasetDict
from datasets.table import Table, list_table_cache_files
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.trainer_pt_utils import DistributedSamplerWithLoop, get_length_grouped_indices
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler as DistributedLengthGroupedSamplerOri
from transformers.trainer_pt_utils import LengthGroupedSampler as LengthGroupedSamplerOri
# from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer, 
    SequentialDistributedSampler, 
    nested_concat,
    )
from transformers import EarlyStoppingCallback

from transformers.utils import logging
from transformers.trainer_utils import denumpify_detensorize, TrainOutput
from training.sampler import sample_by_bald_class_easiness
from training.trainer_base import BaseTrainer
#2024.01.18 pytorch_model.bin to model.safetensor change
from safetensors.torch import load_model, save_model, load_file
import random

random_seed = 42

# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# #np.random.seed(random_seed)
# random.seed(random_seed)


logger = logging.get_logger('Self-training')

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

NEW_WEIGHTS_NAME = "model.safetensors"


class DatasetK(Dataset):
    def __init__(
            self,
            arrow_table: Table,
            info: Optional[DatasetInfo] = None,
            split: Optional[NamedSplit] = None,
            indices_table: Optional[Table] = None,
            fingerprint: Optional[str] = None,
    ):
        self.custom_cache_files = None
        super(DatasetK, self).__init__(arrow_table, info, split, indices_table, fingerprint)


    @property
    def cache_files(self) -> List[dict]:
        """The cache files containing the Apache Arrow table backing the dataset."""
        if self.custom_cache_files:
            return self.custom_cache_files
        cache_files = list_table_cache_files(self._data)
        if self._indices is not None:
            cache_files += list_table_cache_files(self._indices)
        return [{"filename": cache_filename} for cache_filename in cache_files]

    def set_cache_files(self, custom_cache_files):
        self.custom_cache_files = custom_cache_files

def get_class_balanced_loss_weight(samples_per_cls, no_of_classes, beta = 0.99):
    
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    assert len(weights) == len(samples_per_cls)
    return weights
# add by wjn
# revise by ljh
def random_sampling(raw_datasets, num_examples_per_label: Optional[int]=16, least_num=10):
    number = [i for i in range(100)]
    
    np.random.choice(number)

    label_list = raw_datasets["label"] # [0, 1, 0, 0, ...]
    label_dict = dict()
    # 记录每个label对应的样本索引
    for ei, label in enumerate(label_list):
        if label not in label_dict.keys():
            label_dict[label] = list()
        label_dict[label].append(ei)

    # 对于每个类别，随机采样k个样本
    few_example_ids = list()
    for label, eid_list in label_dict.items():
        # examples = deepcopy(eid_list)
        # shuffle(examples)
        idxs = np.random.choice(len(eid_list), size=least_num, replace=False)
        selected_eids = [eid_list[i] for i in idxs]
        few_example_ids.extend(selected_eids)

    remain_examples_num= (num_examples_per_label-least_num)*len(label_dict.keys())
    print('모자란 갯수 : ', remain_examples_num)
    idxs = np.random.choice(len(raw_datasets), size=remain_examples_num, replace=False)
    few_example_ids.extend(idxs)
    
    few_example_ids = list(set(few_example_ids))
    print(len(few_example_ids), '_중복제거샘플')

    return few_example_ids

class TeacherTrainer(BaseTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            test_key: str = "accuracy",
            dataset_name=None,
            random_seed : int = None,
            class_weights: Optional[List[float]] = None
    ):
        super(TeacherTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, class_weights)
        self.predict_dataset = eval_dataset
        self.test_key = test_key
        # if self.args.do_adv:
        #     self.fgm = FGM(self.model)
        # for callback in callbacks:
        #     callback.trainer = self
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        self.global_step_ = 0
        self.dataset_name=dataset_name
        
    

    def mc_evaluate(
        self,
        unlabeled_dataset: Optional[Dataset] = None,
        unlabeled_data_num: int = -1,
        description: str = "Evaluate on Unlabeled Data via MC Dropout Uncertainty Estimation",
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        T: int = 30,
        num_classes: int = 0,
        k_sample : float = 0.0
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        is_sample = True
        if unlabeled_data_num == -1 or unlabeled_data_num >= len(unlabeled_dataset):
            unlabeled_data_num = len(unlabeled_dataset)
            is_sample = False
            print('샘플링 하지 않음')

        else:
            recalled_examples_idx_list = random_sampling(
            raw_datasets=unlabeled_dataset, 
            num_examples_per_label=unlabeled_data_num // num_classes, least_num=10)
            logger.info ("Evaluating uncertainty on {} number of instances sampled from {} unlabeled instances".format(unlabeled_data_num, unlabeled_dataset)) 
            unlabeled_dataset = unlabeled_dataset.select(recalled_examples_idx_list)
            unlabeled_data_num = len(unlabeled_dataset)
            print('샘플링 한다.')

        # else:
        #     logger.info ("Evaluating uncertainty on {} number of instances sampled from {} unlabeled instances".format(unlabeled_data_num, unlabeled_dataset)) 
        #     indices = np.random.choice(len(unlabeled_dataset), unlabeled_data_num, replace=False)
        #     unlabeled_dataset = unlabeled_dataset.select(indices)
        #     unlabeled_data_num = len(unlabeled_dataset)           

        # else:
        #     if self.dataset_name in ["ecommerce", "ecommerce_cate", "ecommerce_cate_top"]:
        #         is_sample=False
        #         logger.info(f"***** mc_evaulate_dataset_name : {self.dataset_name} & is_sample : {is_sample} *****")
        
        # if is_sample:
        #     recalled_examples_idx_list = random_sampling(
        #         raw_datasets=unlabeled_dataset, 
        #         num_examples_per_label=unlabeled_data_num // num_classes
        #     )
        #     unlabeled_dataset = unlabeled_dataset.select(recalled_examples_idx_list)
        #     unlabeled_data_num = len(unlabeled_dataset)

        unlabeled_dataloader = self.get_eval_dataloader(unlabeled_dataset)
        model = self._wrap_model(self.model, training=True, dataloader=unlabeled_dataloader) # reset training to True

        batch_size = unlabeled_dataloader.batch_size
        # unlabeled_data_num = self.num_examples(unlabeled_dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {unlabeled_data_num}")
        logger.info(f"  Batch size = {batch_size}")

        # world_size = max(1, args.world_size)
        
        # if not prediction_loss_only:
        #     # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
        #     # a batch size to the sampler)
        #     make_multiple_of = None
        #     if hasattr(unlabeled_dataloader, "sampler") and isinstance(unlabeled_dataloader.sampler, SequentialDistributedSampler):
        #         make_multiple_of = unlabeled_dataloader.sampler.batch_size

        model.train() # 开启train模式，允许模型进行Dropout

        if args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = unlabeled_dataloader

        # y_T = np.zeros((T, unlabeled_data_num, num_classes))
        y_T = list()
        label = []
        for i in tqdm(range(T)):
            y_pred = []

            for step, inputs in enumerate(unlabeled_dataloader):
                if i == 1:
                    label.extend(inputs['labels'])
                _, logits, __ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                y_pred.extend(logits.detach().cpu().numpy().tolist())
            # print("y_pred.shape=", torch.Tensor(y_pred).shape) # [n, num_class]
            predict_proba = torch.softmax(torch.Tensor(y_pred).to(logits.device), -1)
            k_sample_num = -1 * round(num_classes * k_sample)

            if k_sample != 0.0:
                print('변경전 0 : ', predict_proba[0])
                print('변경전 1 : ', predict_proba[1])
                predict_proba = predict_proba.detach().cpu().numpy().tolist()

                if abs(k_sample_num) < 2:
                    y_T.append(predict_proba.detach().cpu().numpy().tolist())
                    
                else:
                    top_indices = np.argsort(predict_proba)[:, :k_sample_num]
                    
                    for j in range(len(top_indices)):
                        predict_proba[j] = np.where(np.isin(np.arange(num_classes), top_indices[j]), 0, predict_proba[j])

                        row_sum = np.sum(predict_proba[j])
                        if j < 2:
                            print('변경 중간 0 : ', predict_proba[j])
                            print('row_sum : ', row_sum)
                        predict_proba[j] = predict_proba[j] / row_sum
                    print('변경후 0 : ', predict_proba[0])
                    print('변경후 1 : ', predict_proba[1])
                    print("길이 : ", len(predict_proba))
                    y_T.append(predict_proba)
                        
            
            else:
            # print("predict_proba.shape=", predict_proba.shape) # [n, num_class]
            # y_T[i] = predict_proba.detach().cpu().numpy().tolist()
                y_T.append(predict_proba.detach().cpu().numpy().tolist())
        
        y_T = np.array(y_T)
            
        #compute mean
        y_mean = np.mean(y_T, axis=0)
        # print("y_mean.shape=", y_mean.shape) # e.g., (4095, 3) [n, class_num]
        # print("(unlabeled_data_num, num_classes)=", (unlabeled_data_num, num_classes))
        assert y_mean.shape == (unlabeled_data_num, num_classes)

        #compute majority prediction
        y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])
        assert y_pred.shape == (unlabeled_data_num,)

        print(y_pred)
        label = [tensor.item() for tensor in label]
        matching_indices = [index for index, (item1, item2) in enumerate(zip(y_pred, label)) if item1 == item2]
        print(label)
        print('정확도 : ', accuracy_score(label, y_pred))
        #compute variance
        y_var = np.var(y_T, axis=0)
        selected_values = [y_var[i] for i in matching_indices]
        print('일치 분산 : ', np.mean(selected_values))

        unmatching_indices = [index for index, (item1, item2) in enumerate(zip(y_pred, label)) if item1 != item2]
        unselected_values = [y_var[i] for i in unmatching_indices]
        print('불일치 분산 : ', np.mean(unselected_values))        
        
        assert y_var.shape == (unlabeled_data_num, num_classes)

        return unlabeled_dataset, y_mean, y_var, y_pred, y_T, np.array(label)
        # #compute variance
        # y_var = np.var(y_T, axis=0)
        # assert y_var.shape == (unlabeled_data_num, num_classes)

        # return unlabeled_dataset, y_mean, y_var, y_pred, y_T



class RobustTrainer(TeacherTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            test_key: str = "accuracy",
            class_weights: Optional[List[float]] = None
    ):
        super(RobustTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, class_weights)
        self.predict_dataset = eval_dataset
        self.test_key = test_key
        # if self.args.do_adv:
        #     self.fgm = FGM(self.model)
        # for callback in callbacks:
        #     callback.trainer = self
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        self.global_step_ = 0
    
    def robust_train(self):
        pass



class SelfTrainer(object):
    def __init__(
        self, 
        teacher_base_model: torch.nn.Module,
        student_base_model: torch.nn.Module,
        training_args,
        semi_training_args,
        train_dataset: Optional[Dataset]=None,
        unlabeled_dataset: Optional[Dataset]=None,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=None,
        teacher_data_collator=None,
        student_data_collator=None,
        test_key=None,
        task_type="cls",
        num_classes=0,
        dataset_name=None
    ) -> None:

        logger.info("This is a Self-trainer.")
        
        self.teacher_base_model = teacher_base_model
        self.student_base_model = student_base_model
        self.training_args = training_args
        self.semi_training_args = semi_training_args
        self.train_dataset = train_dataset.shuffle()
        self.unlabeled_dataset = unlabeled_dataset.shuffle()
        self.eval_dataset = eval_dataset.shuffle()
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.teacher_data_collator = teacher_data_collator
        self.student_data_collator = student_data_collator
        self.test_key = test_key
        self.task_type = task_type
        self.num_classes = num_classes

        # self.set_teacher_trainer()
        # self.set_student_trainer()
        self.training_args.per_device_train_batch_size = self.semi_training_args.unlabeled_data_batch_size
        self.teacher_training_epoch = self.semi_training_args.teacher_training_epoch # 最初teacher模型在labeled data上训练的epoch数
        self.teacher_tuning_epoch = self.semi_training_args.teacher_tuning_epoch # 每一轮Self-training时，teacher模型继续在labeled data上tune的epoch数
        self.student_training_epoch = self.semi_training_args.student_training_epoch # 每一轮Self-training时，student模型在pseudo-labeled data上训练的epoch数
        self.self_training_epoch = self.semi_training_args.self_training_epoch # Self-training迭代数
        self.unlabeled_data_num = self.semi_training_args.unlabeled_data_num # self-training每轮迭代时，首先挑选一部分用于计算MC dropout uncertainty。-1表示全部计算uncertainty
        self.pseudo_sample_num_or_ratio = self.semi_training_args.pseudo_sample_num_or_ratio # MC dropout后，从所有计算过uncertainty的unlabeled data上采样的样本比例/数量
        self.student_learning_rate = self.semi_training_args.student_learning_rate
        self.student_pre_seq_len = self.semi_training_args.student_pre_seq_len
        self.output_dir = self.training_args.output_dir
        self.alpha = self.semi_training_args.alpha
        self.dataset_name = dataset_name
        self.cb_loss_beta = self.semi_training_args.cb_loss_beta
        self.cb_loss = self.semi_training_args.cb_loss
        self.use_prompt = self.semi_training_args.use_pe
        self.active_learning = self.semi_training_args.active_learning
        self.active_number = self.semi_training_args.active_number
        self.uncert = self.semi_training_args.uncert
        self.up_scale = self.semi_training_args.up_scale
        self.phce_t = self.semi_training_args.phce_t

    def get_teacher_trainer(
        self, 
        base_model: torch.nn.Module, 
        num_train_epochs: int,
        output_dir: str = None,
        class_weights: Optional[List[float]] = None
        ):
        training_args = self.training_args
        training_args.num_train_epochs = num_train_epochs
        if output_dir is not None:
            training_args.output_dir = output_dir
        # 初始化Teacher训练器
        teacher_trainer = TeacherTrainer(
            model=base_model,
            args=training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.teacher_data_collator,
            test_key=self.test_key,
            dataset_name=self.dataset_name,
            class_weights=class_weights,
            # callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
        )
        return teacher_trainer
    def predict_data(self, trainer, predict_dataset=None, log_file_path=None):
        if predict_dataset is None:
            logger.info("No dataset is available for testing")
            return
    
        if log_file_path is None:
            logger.info("No log file path provided")
            return
    
        with open(log_file_path, "w") as f:
            if isinstance(predict_dataset, dict):
                for dataset_name, d in predict_dataset.items():
                    logger.info("*** Predict: %s ***" % dataset_name)
                    predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
                    predictions = np.argmax(predictions, axis=2)
    
                    predicted_labels = predictions.tolist()
    
                    trainer.log_metrics("predict", metrics)
                    trainer.save_metrics("predict", metrics)
    
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write("Metrics:\n")
                    f.write(f"{metrics}\n\n")
    
            else:
                logger.info("*** Predict ***")
                predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
                predictions = np.argmax(predictions, axis=1)
    
                predicted_labels = predictions.tolist()
    
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics)
    
                f.write("Metrics:\n")
                f.write(f"{metrics}\n\n")
    
            f1_score_macro = f1_score(predict_dataset['label'], predicted_labels, average="macro")
            recall_macro = recall_score(predict_dataset['label'], predicted_labels, average="macro")
            precision_macro = precision_score(predict_dataset['label'], predicted_labels, average="macro")
    
            f.write(f"f1_score_macro: {f1_score_macro}\n")
            f.write(f"recall_macro: {recall_macro}\n")
            f.write(f"precision_macro: {precision_macro}\n")
    
            f1_score_micro = f1_score(predict_dataset['label'], predicted_labels, average="micro")
            recall_micro = recall_score(predict_dataset['label'], predicted_labels, average="micro")
            precision_micro = precision_score(predict_dataset['label'], predicted_labels, average="micro")
    
            f.write(f"f1_score_micro: {f1_score_micro}\n")
            f.write(f"recall_micro: {recall_micro}\n")
            f.write(f"precision_micro: {precision_micro}\n")
    
            f1_score_weighted = f1_score(predict_dataset['label'], predicted_labels, average="weighted")
            recall_weighted = recall_score(predict_dataset['label'], predicted_labels, average="weighted")
            precision_weighted = precision_score(predict_dataset['label'], predicted_labels, average="weighted")
    
            f.write(f"f1_score_weighted: {f1_score_weighted}\n")
            f.write(f"recall_weighted: {recall_weighted}\n")
            f.write(f"precision_weighted: {precision_weighted}\n")
    
            accuracy = accuracy_score(predict_dataset['label'], predicted_labels)
            f.write(f"accuracy_score: {accuracy}\n")

    
    def get_student_trainer(
        self, 
        base_model: torch.nn.Module, 
        num_train_epochs: int, 
        student_learning_rate: float,
        pseudo_labeled_dataset: Optional[Dataset] = None, 
        output_dir: str = None,
        class_weights: Optional[List[float]] = None
        ):
        training_args = self.training_args
        training_args.num_train_epochs = num_train_epochs
        training_args.learning_rate = student_learning_rate
        if output_dir is not None:
            training_args.output_dir = output_dir
        # 初始化Student训练器
        student_trainer = RobustTrainer(
            model=base_model,
            args=training_args,
            train_dataset=pseudo_labeled_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.student_data_collator,
            test_key=self.test_key,
            class_weights=class_weights,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=15)]
        )
        return student_trainer

    def freeze_backbone(self, model: torch.nn.Module, use_pe: bool=False):
        try:
            model.freeze_backbone(use_pe=use_pe)
        except:
            pass
        return model

    def cb_loss_weight(self, model, class_weights=None):
        model.get_cb_loss_weight(class_weights=class_weights)
    
    def train(self, resume_from_checkpoint=None):
        if not os.path.exists(os.path.join(self.output_dir, "iteration")):
            os.makedirs(os.path.join(self.output_dir, "iteration"))

        teacher_model = self.teacher_base_model
        teacher_model = self.freeze_backbone(teacher_model, use_pe=False)
        teacher_trainer: TeacherTrainer = self.get_teacher_trainer(base_model=teacher_model, num_train_epochs=self.teacher_training_epoch)
        logger.info("*"*80)
        logger.info("* teacher model train dataset shape : {} *".format(teacher_trainer.train_dataset.shape))
        #print(teacher_trainer.train_dataset.shape)
        
        # 2024.01.19 CHANGE WEIGHTS_NAME to NEW_WEIGHTS_NAME
        if resume_from_checkpoint is not None and (os.path.isfile(os.path.join(resume_from_checkpoint, NEW_WEIGHTS_NAME)) or os.path.isfile(
            os.path.join(resume_from_checkpoint, NEW_WEIGHTS_NAME))
        ):
            logger.info("*"*80)
            logger.info("* Directly loading the trained teacher model from {} *".format(resume_from_checkpoint))
            logger.info("*"*80)
            print("*"*80)
            logger.info("* Directly loading the trained teacher model from {} *".format(resume_from_checkpoint))
            print("*"*80)
            # 已有teacher模型，直接加载
            teacher_trainer._load_from_checkpoint(resume_from_checkpoint)
        else:

            # 首先对Teacher模型在labeled data上进行full parameter fine-tuning
            logger.info("*"*66)
            logger.info("* Training teacher model over labeled data before self-training. *")
            logger.info("*"*66)
            print("*"*66)
            print("* Training teacher model over labeled data before self-training. *")
            print("*"*66)

            teacher_trainer.train()
            #2024.01.18 코드 수정
            load_model(teacher_model, os.path.join(teacher_trainer.state.best_model_checkpoint, "model.safetensors"))
            #teacher_model.load_state_dict(torch.load(os.path.join(teacher_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
            teacher_trainer.model = teacher_model

        # 原始的训练结果
        metrics = teacher_trainer.evaluate()
        convention_result = metrics["eval_{}".format(self.test_key)]

        logger.info("*"*50)
        logger.info("* Conventional fine-tuning metric: {}. *".format(convention_result))
        logger.info("*"*50)
        print("*"*50)
        print("* Conventional fine-tuning metric: {}. *".format(convention_result))
        print("*"*50)

        logger.info("*"*30)
        logger.info("* Starting Active Self-training ... *")
        logger.info("*"*30)
        print("*"*30)
        print("* Starting Active Self-training ... *")
        print("*"*30)

        best_test_metric = None
        best_self_training_iteration = None
        best_teacher_model = None

        self.predict_data(teacher_trainer, self.eval_dataset, os.path.join(self.output_dir, "total_metrics"))

        # 多轮Teacher-Student迭代训练

        plus_pseudo_sample_num = self.pseudo_sample_num_or_ratio
        for iter in range(self.self_training_epoch):
            if self.active_learning : 
        
                logger.info("*"*34)
                logger.info("* Active-learning {}-th iteration *".format(iter))
                logger.info("*"*34)
                print("*"*34)
                print("* Active-learning {}-th iteration *".format(iter))
                print("*"*34)
    
    
                # 获得Teacher模型在测试集上的效果
                if iter > 0:
                    teacher_trainer.model = teacher_model
                    metrics = teacher_trainer.evaluate()
                    # print("metrics=", metrics)
                
                '''
                e.g., {'eval_loss': 0.6926815509796143, 'eval_accuracy': 0.5234657039711191, 'eval_runtime': 0.7267, 'eval_samples_per_second': 381.161, 'eval_steps_per_second': 48.161, 'epoch': 1.0}
                '''
                logger.info("*"*60)
                logger.info("* The testing result of teacher model is {} result: {} *".format(self.test_key, metrics["eval_{}".format(self.test_key)]))
                logger.info("*"*60)
                print("*"*60)
                print("* The testing result of teacher model is {} result: {} *".format(self.test_key, metrics["eval_{}".format(self.test_key)]))
                print("*"*60)
    
                if best_test_metric is None or best_test_metric < metrics["eval_{}".format(self.test_key)]:
                    best_test_metric = metrics["eval_{}".format(self.test_key)]
                    best_self_training_iteration = iter
                    best_teacher_model = teacher_model
                    logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                    logger.info("The best teacher model testing result is {}.".format(best_test_metric))
                    print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                    print("The best teacher model testing result is {}.".format(best_test_metric))
    
                
                # Teacher模型在unlabeled data上获取pseudo-labeled data，并根据uncertainty estimation进行采样
                logger.info("*"*72)
                logger.info("Active_learning for Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
                logger.info("*"*72)
                print("*"*72)
                print("Active_learning for Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
                print("*"*72)
    
                unlabeled_dataset, y_mean, y_var, y_pred, y_T, true_label = teacher_trainer.mc_evaluate(
                    unlabeled_dataset=self.unlabeled_dataset, 
                    unlabeled_data_num=self.unlabeled_data_num,
                    T=15, 
                    num_classes=self.num_classes
                    )
                
                logger.info("*"*42)
                logger.info("* Sampling reliable pseudo-labeled data. *")
                logger.info("*"*42)
                print("*"*42)
                print("* Sampling reliable pseudo-labeled data. *")
                print("*"*42)
                # pseudo_labeled_input, np.array(y_s), np.array(w_s), active_labeled_input, np.array(active_y_s), np.array(active_w_s), active_X_idxs
                X_batch, y_batch, w_batch, active_X_batch, active_y_batch, active_w_batch, X_idxs = sample_by_bald_class_easiness(
                    tokenizer=self.tokenizer, 
                    X=unlabeled_dataset, 
                    y_mean=y_mean, 
                    y_var=y_var, 
                    y=y_pred, 
                    num_samples=int(y_pred.shape[0] * self.pseudo_sample_num_or_ratio) if self.pseudo_sample_num_or_ratio <= 1.0 else int(self.pseudo_sample_num_or_ratio), 
                    num_classes=self.num_classes, 
                    y_T=y_T,
                    alpha=self.alpha,
                    cb_loss=self.cb_loss,
                    true_label = true_label,
                    active_learning = self.active_learning,
                    active_number = self.active_number,
                    uncert = self.uncert,
                    up_scale = self.up_scale)

                # Active_learning을 통해 레이블링된 데이터는 언레이블링 데이터에서 제외 처리.
                print('제외 전 언레이블 데이터 : ', len(self.unlabeled_dataset))
                check_number = len(self.unlabeled_dataset)
                self.unlabeled_dataset = self.unlabeled_dataset.filter(lambda example: example['idx'] not in X_idxs)
                print('제외 후 언레이블 데이터 : ', len(self.unlabeled_dataset))
                print('차이 체크 : ', check_number - len(self.unlabeled_dataset), '정상 유무 : ', check_number - len(self.unlabeled_dataset) == self.active_number * self.num_classes)
                
                print("{} : 클래스별 샘플링 갯수 모음".format(np.bincount(y_batch)))
    
                active_labeled_examples = active_X_batch
                active_labeled_examples["label"] = active_y_batch  
                active_labeled_dataset = DatasetK.from_dict(active_labeled_examples)

                for i in range(len(active_labeled_dataset)):
                    tmp_dataset=active_labeled_dataset[i]
                    self.train_dataset = self.train_dataset.add_item(tmp_dataset)
                    print(i,"_번쨰_train_dataset_length : {}".format(len(self.train_dataset)))

            
                if self.semi_training_args.confidence:
                    logger.info("* Confidence Learning Operation and conf_alpha : {} *".format(self.semi_training_args.conf_alpha))
                    print("* Confidence Learning Operation and conf_alpha : {} *".format(self.semi_training_args.conf_alpha))
                    X_conf = -np.log(w_batch+1e-10)*self.semi_training_args.conf_alpha
                    pseudo_labeled_examples = X_batch
                    pseudo_labeled_examples["label"] = y_batch
                    pseudo_labeled_examples["t"] = self.phce_t

                    
                else:
                    pseudo_labeled_examples = X_batch
                    pseudo_labeled_examples["label"] = y_batch               
                
                # 生成pseudo-labeled dataset
                # pseudo_labeled_dataset = DatasetDict()
                pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)
                
                for i in range(len(self.train_dataset)):
                    tmp_dataset=self.train_dataset[i]
    
                    if self.semi_training_args.confidence:
                        labeled_data_conf = -np.log(1e-10)*self.semi_training_args.conf_alpha
                        tmp_dataset["t"] = self.phce_t
    
                    pseudo_labeled_dataset = pseudo_labeled_dataset.add_item(tmp_dataset)
                
    
                # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
                logger.info("*"*56)
                logger.info("* Training a new student model on active-labeled data. *")
                logger.info("*"*56)
                print("*"*56)
                print("* Training a new student model on active-labeled data. *")
                print("*"*56)

                if iter > 0 :
                    student_model = teacher_model

                else:
                    if self.use_prompt:
                        print(iter, '_ 프롬프트 튜닝을 위한 초기화')
                        student_model = self.student_base_model
                        student_model = self.freeze_backbone(student_model, use_pe=True)
                        print(iter, '프롬프트 백본 프리징')
                    else:
                        student_model = teacher_model
    
                student_trainer: RobustTrainer = self.get_student_trainer(
                    base_model=student_model, 
                    num_train_epochs=self.teacher_training_epoch,
                    student_learning_rate=self.student_learning_rate,
                    pseudo_labeled_dataset=self.train_dataset,
                    output_dir=os.path.join(self.output_dir, "iteration", "active_student_iter_{}".format(iter))
                )
                student_trainer.train()
                load_model(student_model, os.path.join(student_trainer.state.best_model_checkpoint, "model.safetensors"))
                #student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
    
                # 将Student模型参数赋给Teacher，作为下一轮训练的Teacher初始化
                logger.info("*"*64)
                logger.info("* Initializing a new teacher model from trained student model. *")
                logger.info("*"*64)
                print("*"*64)
                print("* Initializing a new teacher model from trained student model. *")
                print("*"*64)
                teacher_model = student_model
                # teacher_trainer = student_trainer
                teacher_trainer: TeacherTrainer = self.get_teacher_trainer(
                    base_model=student_model, 
                    num_train_epochs=self.teacher_tuning_epoch, 
                    output_dir=os.path.join(self.output_dir, "iteration", "active_teacher_iter_{}".format(iter))
                )
    
    
                
            
                logger.info("********** Finishing Active-learning **********")
                logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                logger.info("The best teacher model testing result is {}.".format(best_test_metric))
                print("********** Finishing Active-learning **********")
                print("The best teacher model at {}-th Active-learning iteration.".format(best_self_training_iteration))
                print("The best teacher model testing result is {}.".format(best_test_metric))
                # self.predict_data(teacher_trainer, self.eval_dataset, os.path.join(self.output_dir, "total_metrics_last"))

            else:
        
            
                logger.info("*"*34)
                logger.info("* Self-training {}-th iteration *".format(iter))
                logger.info("*"*34)
                print("*"*34)
                print("* Self-training {}-th iteration *".format(iter))
                print("*"*34)
    
    
                # 获得Teacher模型在测试集上的效果
                if iter > 0:
                    teacher_trainer.model = teacher_model
                    metrics = teacher_trainer.evaluate()
                    # print("metrics=", metrics)
                
                '''
                e.g., {'eval_loss': 0.6926815509796143, 'eval_accuracy': 0.5234657039711191, 'eval_runtime': 0.7267, 'eval_samples_per_second': 381.161, 'eval_steps_per_second': 48.161, 'epoch': 1.0}
                '''
                logger.info("*"*60)
                logger.info("* The testing result of teacher model is {} result: {} *".format(self.test_key, metrics["eval_{}".format(self.test_key)]))
                logger.info("*"*60)
                print("*"*60)
                print("* The testing result of teacher model is {} result: {} *".format(self.test_key, metrics["eval_{}".format(self.test_key)]))
                print("*"*60)
    
                if best_test_metric is None or best_test_metric < metrics["eval_{}".format(self.test_key)]:
                    best_test_metric = metrics["eval_{}".format(self.test_key)]
                    best_self_training_iteration = iter
                    best_teacher_model = teacher_model
                    logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                    logger.info("The best teacher model testing result is {}.".format(best_test_metric))
                    print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                    print("The best teacher model testing result is {}.".format(best_test_metric))
                
    
                # if iter == self.self_training_epoch - 1:
                #     self.unlabeled_data_num = 80000
                #     self.pseudo_sample_num_or_ratio = int(self.unlabeled_data_num * 0.5)
                #     T = 10
    
                # else :
                #     T = 16
                    
                    
                #     break
                
                # Teacher模型在unlabeled data上获取pseudo-labeled data，并根据uncertainty estimation进行采样
                logger.info("*"*72)
                logger.info("Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
                logger.info("*"*72)
                print("*"*72)
                print("Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
                print("*"*72)
    
                unlabeled_dataset, y_mean, y_var, y_pred, y_T, true_label = teacher_trainer.mc_evaluate(
                    unlabeled_dataset=self.unlabeled_dataset, 
                    unlabeled_data_num=self.unlabeled_data_num,
                    T=15, 
                    num_classes=self.num_classes
                    )
                
                logger.info("*"*42)
                logger.info("* Sampling reliable pseudo-labeled data. *")
                logger.info("*"*42)
                print("*"*42)
                print("* Sampling reliable pseudo-labeled data. *")
                print("*"*42)
                
                X_batch, y_batch, w_batch, _, _, _, _, = sample_by_bald_class_easiness(
                    tokenizer=self.tokenizer, 
                    X=unlabeled_dataset, 
                    y_mean=y_mean, 
                    y_var=y_var, 
                    y=y_pred, 
                    num_samples=int(y_pred.shape[0] * self.pseudo_sample_num_or_ratio) if self.pseudo_sample_num_or_ratio <= 1.0 else int(self.pseudo_sample_num_or_ratio), 
                    num_classes=self.num_classes, 
                    y_T=y_T,
                    alpha=self.alpha,
                    cb_loss=self.cb_loss,
                    true_label = true_label,
                    uncert = self.uncert,
                    up_scale = self.up_scale)
    
                #num_samples = int(num_samples * 1.2)
                #self.unlabeled_data_num = int(self.unlabeled_data_num * 1.1)
    
                #print(w_batch, len(w_batch))
                print("{} : 클래스별 샘플링 갯수 모음".format(np.bincount(y_batch) + (len(self.train_dataset) / self.num_classes)))
    
                
                print(iter, " 번째 self-training 변경 전 pseudo_sample_num_or_ratio : ", self.pseudo_sample_num_or_ratio)
                self.pseudo_sample_num_or_ratio += plus_pseudo_sample_num
                print(iter, " 번째 self-training 변경 후 pseudo_sample_num_or_ratio : ", self.pseudo_sample_num_or_ratio)
                
                if iter == 2:
                    print(iter, " 번째 self-training 변경 전 unlabeled_data_num : ", self.unlabeled_data_num)
                    self.unlabeled_data_num += self.unlabeled_data_num
                    print(iter, " 번째 self-training 변경 후 unlabeled_data_num : ", self.unlabeled_data_num)
    
                # if self.cb_loss:
                #     logger.info("Check Balanced_Loss : {}".format(self.cb_loss))
                #     logger.info("Class Balanced_Loss_beta : {}".format(self.cb_loss_beta))
                #     class_count=np.bincount(y_batch) + (len(self.train_dataset) // self.num_classes)
                #     class_weights=get_class_balanced_loss_weight(class_count, self.num_classes, beta = self.cb_loss_beta)
                    
      
                # add by ljh(copy UST)
                if self.semi_training_args.confidence:
                    logger.info("* Confidence Learning Operation and conf_alpha : {} *".format(self.semi_training_args.conf_alpha))
                    print("* Confidence Learning Operation and conf_alpha : {} *".format(self.semi_training_args.conf_alpha))
                    #X_conf = -np.log(w_batch+1e-10)*self.semi_training_args.conf_alpha
                    pseudo_labeled_examples = X_batch
                    pseudo_labeled_examples["label"] = y_batch
                    pseudo_labeled_examples["t"] = np.ones(len(y_batch))*self.phce_t
                    # if self.cb_loss:
                    #     pseudo_labeled_examples["class_weights"] = np.repeat([class_weights], len(y_batch), axis=0)
                    
                else:
                    pseudo_labeled_examples = X_batch
                    pseudo_labeled_examples["label"] = y_batch               
                
                # 生成pseudo-labeled dataset
                # pseudo_labeled_dataset = DatasetDict()
                pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)
                
                for i in range(len(self.train_dataset)):
                    tmp_dataset=self.train_dataset[i]
    
                    if self.semi_training_args.confidence:
                        #labeled_data_conf = -np.log(1e-10)*self.semi_training_args.conf_alpha
                        tmp_dataset["t"] = self.phce_t
                        # tmp_dataset["class_weights"] = class_weights
                        
                    # if not self.semi_training_args.confidence:
                    #     tmp_dataset["weight"] = 1.0
                        
                    # else:
                    #     labeled_data_conf = -np.log(1e-10)*self.semi_training_args.conf_alpha
                    #     tmp_dataset["weight"] = labeled_data_conf
    
                    pseudo_labeled_dataset = pseudo_labeled_dataset.add_item(tmp_dataset)
    
                # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
                logger.info("*"*56)
                logger.info("* Training a new student model on pseudo-labeled data. *")
                logger.info("*"*56)
                print("*"*56)
                print("* Training a new student model on pseudo-labeled data. *")
                print("*"*56)
    
                if self.use_prompt:
                    print("USE_PROMPT AND STUDENT_MODEL SELF_TRAINING NOT INITIALIZE: {} ITERATION".format(iter))
                    if iter >= 0:
                        student_model = teacher_model
                    # student_model = self.student_base_model
                    # student_model = self.freeze_backbone(student_model, use_pe=True)
    
                else:
                    # if iter == 0:
                    print("NOT_USE_PROMPT AND STUDENT_MODEL INITIALIZE : {} ITERATION".format(iter))
                    student_model = teacher_model
                        # student_model = self.student_base_model
                        # student_model = self.freeze_backbone(student_model, use_pe=False)
                    
    
                    # else:
                    #     print("NOT_USE_PROMPT AND STUDENT_MODEL CONTINUOUS : {} ITERATION".format(iter))
                        
    
                student_trainer: RobustTrainer = self.get_student_trainer(
                    base_model=student_model, 
                    num_train_epochs=self.student_training_epoch,
                    student_learning_rate=self.student_learning_rate,
                    pseudo_labeled_dataset=pseudo_labeled_dataset,
                    output_dir=os.path.join(self.output_dir, "iteration", "student_iter_{}".format(iter))
                )
                student_trainer.train()
                #2024.01.18 코드 수정
                load_model(student_model, os.path.join(student_trainer.state.best_model_checkpoint, "model.safetensors"))
                #student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
    
                # 将Student模型参数赋给Teacher，作为下一轮训练的Teacher初始化
                logger.info("*"*64)
                logger.info("* Initializing a new teacher model from trained student model. *")
                logger.info("*"*64)
                print("*"*64)
                print("* Initializing a new teacher model from trained student model. *")
                print("*"*64)
                teacher_model = student_model
                # teacher_trainer = student_trainer
                teacher_trainer: TeacherTrainer = self.get_teacher_trainer(
                    base_model=student_model, 
                    num_train_epochs=self.teacher_tuning_epoch, 
                    output_dir=os.path.join(self.output_dir, "iteration", "teacher_iter_{}".format(iter))
                )
            
            self.predict_data(teacher_trainer, self.eval_dataset, os.path.join(self.output_dir, "total_metrics_last"))
    
            logger.info("********** Finishing Self-training **********")
            logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
            logger.info("The best teacher model testing result is {}.".format(best_test_metric))
            print("********** Finishing Self-training **********")
            print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
            print("The best teacher model testing result is {}.".format(best_test_metric))

        
        # 根据当前最好的Teacher模型，在全部的unlabeled data上打伪标签，并进行mc dropout（样本数量最多不超过50000）
        if self.semi_training_args.post_student_train:
            
            logger.info("********** Post training **********")
            print("********** Post training **********")

            teacher_trainer: TeacherTrainer = self.get_teacher_trainer(
                base_model=best_teacher_model, 
                num_train_epochs=self.teacher_tuning_epoch, 
                output_dir=os.path.join(self.output_dir, "teacher_iter_post")
            )

            unlabeled_dataset, y_mean, y_var, y_pred, y_T, true_label = teacher_trainer.mc_evaluate(
                unlabeled_dataset=self.unlabeled_dataset, 
                unlabeled_data_num=80000,
                T=10, 
                num_classes=self.num_classes
                )
            
            post_sample_num = int(self.num_classes * 500)
            
            X_batch, y_batch, w_batch, _ = sample_by_bald_class_easiness(
                tokenizer=self.tokenizer, 
                X=unlabeled_dataset, 
                y_mean=y_mean, 
                y_var=y_var, 
                y=y_pred, 
                num_samples=post_sample_num, 
                num_classes=self.num_classes, 
                y_T=y_T,
                alpha=self.alpha,
                cb_loss=self.cb_loss,
                true_label = true_label,
                uncert = self.uncert,
                up_scale = self.up_scale)
            
            # add by ljh(copy UST)
            # if self.semi_training_args.confidence:
            #     logger.info("* Confidence Learning Not Operation*")
            #     X_conf = np.ones(len(X_batch['input_ids']))

            # else :    
            #     logger.info("* Confidence Learning Operation and conf_alpha : {} *".format(self.semi_training_args.conf_alpha))
            #     X_conf = -np.log(w_batch+1e-10)*self.semi_training_args.conf_alpha
            
            #print(w_batch, len(w_batch))
            print("{} : 클래스별 샘플링 갯수 모음".format(np.bincount(y_batch) + (len(self.train_dataset) / self.num_classes)))


            # if self.cb_loss:
            #     logger.info("Check Balanced_Loss : {}".format(self.cb_loss))
            #     logger.info("Class Balanced_Loss_beta : {}".format(self.cb_loss_beta))
                
            #     class_count=np.bincount(y_batch) + (len(self.train_dataset) // self.num_classes)
            #     class_weights=get_class_balanced_loss_weight(class_count, self.num_classes, beta = self.cb_loss_beta)
            
            if self.semi_training_args.confidence:
                logger.info("* Confidence Learning Operation and conf_alpha : {} *".format(self.semi_training_args.conf_alpha))
                X_conf = -np.log(w_batch+1e-10)*self.semi_training_args.conf_alpha
                pseudo_labeled_examples = X_batch
                pseudo_labeled_examples["label"] = y_batch
                pseudo_labeled_examples["t"] = self.phce_t
                #pseudo_labeled_examples["class_weights"] = np.repeat([class_weights], len(y_batch), axis=0)
            else:
                pseudo_labeled_examples = X_batch
                pseudo_labeled_examples["label"] = y_batch               
            
            # 生成pseudo-labeled dataset
            # pseudo_labeled_dataset = DatasetDict()
            pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)
            
            for i in range(len(self.train_dataset)):
                tmp_dataset=self.train_dataset[i]

                if self.semi_training_args.confidence:
                    labeled_data_conf = -np.log(1e-10)*self.semi_training_args.conf_alpha
                    tmp_dataset["t"] = self.phce_t
                    # tmp_dataset["class_weights"] = class_weights
                    
                # if not self.semi_training_args.confidence:
                #     tmp_dataset["weight"] = 1.0
                    
                # else:
                #     labeled_data_conf = -np.log(1e-10)*self.semi_training_args.conf_alpha
                #     tmp_dataset["weight"] = labeled_data_conf

                pseudo_labeled_dataset = pseudo_labeled_dataset.add_item(tmp_dataset)

            # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
            logger.info("*"*56)
            logger.info("* Training a new student model on pseudo-labeled data. *")
            logger.info("*"*56)
            print("*"*56)
            print("* Training a new student model on pseudo-labeled data. *")
            print("*"*56)
            
            if self.use_prompt:
                print("USE_PROMPT AND STUDENT_MODEL INITIALIZE")
                student_model = self.student_base_model
                student_model = self.freeze_backbone(student_model, use_pe=True)

            else:
                # if iter == 0:
                print("NOT_USE_PROMPT AND STUDENT_MODEL INITIALIZE : {} ITERATION".format(iter))
                student_model = teacher_model
                    # student_model = self.student_base_model
                    # student_model = self.freeze_backbone(student_model, use_pe=False)

            student_trainer: RobustTrainer = self.get_student_trainer(
                base_model=student_model, 
                num_train_epochs=self.student_training_epoch if len(pseudo_labeled_dataset) <= 4096 else int(self.student_training_epoch / 2),
                student_learning_rate=self.student_learning_rate,
                pseudo_labeled_dataset=pseudo_labeled_dataset,
                output_dir=os.path.join(self.output_dir, "student_iter_{}".format(iter))
            )

                
            student_trainer.train()
            # 2024.01.18 코드 수정
            load_model(student_model, os.path.join(student_trainer.state.best_model_checkpoint, "model.safetensors"))
            #student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))

            metrics = student_trainer.evaluate()
            post_metric = metrics["eval_{}".format(self.test_key)]


        print("*"*68)
        print("Finishing all the processes, the results are shown in the following:")
        print("Conventional fine-tuning {} metric: {}".format(self.test_key, convention_result))
        print("Best self-training {} metric: {}".format(self.test_key, best_test_metric))
        if self.semi_training_args.post_student_train:
            print("Post training {} metric: {}".format(self.test_key, post_metric))
        print("*"*68)
        self.predict_data(student_trainer, self.eval_dataset, os.path.join(self.output_dir, "total_metrics_last"))
        
        return TrainOutput(teacher_trainer.state.global_step, 0.0, metrics)
