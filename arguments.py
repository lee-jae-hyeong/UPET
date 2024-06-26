from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import *


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """
    # 
    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    # add by wjn
    num_examples_per_label: Optional[int] = field(
        default=None,
        metadata={
            "help": "Randomly sampling k-shot examples for each label "
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        })
    
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_pe: bool = field(
        default=False,
        metadata={
            "help": "Whether to use parameter-efficient settings. If true, that means fix the parameters of the backbone, and only"
            "tune the new initialized modules (e.g., adapter, prefix, ptuning, etc.)"
        }
    )
    head_prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use Head-tuning Prefix (P-tuning v2) during training"
        }
    )
    prompt_prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use Prompt-tuning Prefix (P-tuning v2) during training"
        }
    )
    head_only: bool = field(
        default=False,
        metadata={
            "help": "Will use Head Fine-tuning during training (w/o. any new modules)"
        }
    )
    prompt_only: bool = field(
        default=False,
        metadata={
            "help": "Will use Prompt Fine-tuning during training (w/o. any new modules)"
        }
    )
    head_ptuning: bool = field(
        default=False,
        metadata={
            "help": "Will use Head-tuning P-tuning during training (w. parameter-efficient)"
        }
    )
    prompt_ptuning: bool = field(
        default=False,
        metadata={
            "help": "Will use Prompt-tuning P-tuning during training (w. parameter-efficient)"
        }
    )
    head_adapter: bool = field(
        default=False,
        metadata={
            "help": "Will use Head-tuning Adapter tuning during training (w. parameter-efficient)"
        }
    )
    prompt_adapter: bool = field(
        default=False,
        metadata={
            "help": "Will use Prompt-tuning Adapter tuning during training (w. parameter-efficient)"
        }
    )
    adapter_choice: str = field(
        default="LiST",
        metadata={"help": "The choice of adapter, list, lora, houlsby."},
    )
    adapter_dim: int = field(
        default=128,
        metadata={"help": "The hidden size of adapter. default is 128."},
    )
    pre_seq_len: int = field(
        default=4,
        metadata={
            "help": "The length of prompt"
        }
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    ) 
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        })
    



@dataclass
class SemiSupervisedArguments:
    use_semi: bool = field(
        default=False, metadata={"help": "If true, the training process will be transformed into self-training framework."}
    )
    unlabeled_data_num: int = field(
        default=-1,
        metadata={
            "help": "The total number of unlabeled data. If set -1 means all the training data (expect of few-shot labeled data)"
        }
    )
    unlabeled_data_batch_size: int = field(
        default=16,
        metadata={
            "help": "The number of unlabeled data in one batch."
        }
    )
    pseudo_sample_num_or_ratio: float = field(
        default=0.1,
        metadata={
            "help": "The number / ratio of pseudo-labeled data sampling. For example, if have 1000 unlabeled data, 0.1 / 100 means sampling 100 pseduo-labeled data."
        }
    )
    teacher_training_epoch: int = field(
        default=10,
        metadata={
            "help": "The epoch number of teacher training at the beginning of self-training."
        }
    )
    teacher_tuning_epoch: int = field(
        default=10,
        metadata={
            "help": "The epoch number of teacher tuning in each self-training iteration."
        }
    )
    student_training_epoch: int = field(
        default=16,
        metadata={
            "help": "The epoch number of student training in each self-training iteration."
        }
    )
    student_learning_rate: float = field(
        default=1e-5,
        metadata={
            "help": "The learning rate of student training in each self-training iteration."
        }
    )
    self_training_epoch: int = field(
        default=30,
        metadata={
            "help": "The number of teacher-student iteration ."
        }
    )
    # self-training过程中，unlabeled data会先采样一个子集，因此student模型学习的样本数量只有不到1024个
    # post_student_train设置后，即允许在self-training后，选择最佳的teacher模型，并在全部的unlabeled data上打标和mc dropout
    # 此时student模型将会在更多的样本上训练（样本数量不超过20k）
    post_student_train: bool = field(
        default=False,
        metadata={
            "help": "Whether to train a student model on large pseudo-labeled data after self-training iteration"
        }
    )
    student_pre_seq_len: int = field(
        default=4,
        metadata={
            "help": "The length of prompt"
        }
    )
    alpha : Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Reliable Example Sampling model confiden and certainty ratio"
        }
    )
    confidence : bool = field(
        default=False,
        metadata={
            "help": "Confidence Learning check - Var by MC dropout"
        }
    )
    conf_alpha : Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Confidence Learning Rate"
        }
    )
    cb_loss : bool = field(
        default=False,
        metadata={
            "help": "Class Balance Loss"})
    
    cb_loss_beta : Optional[float] = field(
        default=0.99,
        metadata={
        "help" : "cb_loss ratio"}
    )
    active_learning :  bool = field(
        default=False,
        metadata={
        "help" : "Active_learning"}
    )
    active_number : Optional[int] = field(
        default=16,
        metadata={
        "help" : "Active_learning per sample number"}
    )

    uncert :  bool = field(
        default=True,
        metadata={
        "help" : "Uncertainty_check"})

    up_scale :  bool = field(
        default=True,
        metadata={
        "help" : "Uncertainty_up_scale_check"})

    phce_t :  Optional[float] = field(
        default=1.2,
        metadata={
        "help" : "phce_ce for t parameter"})
    
    c_type: str = field(
        default="BALD",
        metadata={"help": "selection BALD, RES, CONF, entropy, marginal, var."})

@dataclass
class QuestionAnwseringArguments:
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )




def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, SemiSupervisedArguments, QuestionAnwseringArguments))

    args = parser.parse_args_into_dataclasses()

    return args
