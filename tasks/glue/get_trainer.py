import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.glue.dataset import GlueDataset
from training.trainer_base import BaseTrainer
from training.self_trainer import SelfTrainer
from datasets import load_from_disk

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, semi_training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    # add by wjn check if use prompt template
    use_prompt = False
    if model_args.prompt_prefix or model_args.prompt_ptuning or model_args.prompt_adapter or model_args.prompt_only:
        use_prompt = True

    dataset = GlueDataset(tokenizer, data_args, training_args, semi_training_args=semi_training_args, use_prompt=use_prompt)
    
    data_args.label_word_list = None # add by wjn
    if use_prompt:
        data_args.label_word_list = dataset.label_word_list # add by wjn

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    model = get_model(data_args, model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    if data_args.dataset_name == 'ecommerce':
        path = "/content/drive/MyDrive/UPET/ecommerce"
    elif data_args.dataset_name == 'ecommerce_cate':
        path = "/content/drive/MyDrive/UPET/ecommerce_cate"
    elif data_args.dataset_name == 'ecommerce_cate_top':
        path = "/content/drive/MyDrive/UPET/ecommerce_cate_top"
    elif data_args.dataset_name == 'e_cate2':
        path = "/content/drive/MyDrive/UPET/e_cate2"

    if data_args.dataset_name in ["ecommerce", "ecommerce_cate", "ecommerce_cate_top", "e_cate2"]:
        
        raw_datasets = load_from_disk(path)
        new_token = raw_datasets["train"].features["label"].names
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        # new_tokens = set(new_token) - set(tokenizer.vocab.keys())
        # tokenizer.add_tokens(list(new_tokens))
        # model.resize_token_embeddings(len(tokenizer))
    # Initialize our Trainer

    if semi_training_args.use_semi:
        model_args.pre_seq_len = semi_training_args.student_pre_seq_len
        student_model = get_model(data_args, model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

        # if data_args.dataset_name in ["ecommerce", "ecommerce_cate", "ecommerce_cate_top"]:
        #     student_model.resize_token_embeddings(len(tokenizer))

        trainer = SelfTrainer(
            teacher_base_model=model,
            student_base_model=student_model,
            training_args=training_args,
            semi_training_args=semi_training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            unlabeled_dataset=dataset.unlabeled_dataset,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            teacher_data_collator=dataset.data_collator,
            student_data_collator=dataset.data_collator,
            test_key=dataset.test_key,
            task_type="cls",
            num_classes=len(dataset.label2id),
            dataset_name=data_args.dataset_name
        )

        return trainer, None

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key,
    )

    return trainer, None
