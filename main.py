from trainer import Trainer
import json
import argparse
import os
import numpy as np
import random
import torch
from transformers import set_seed


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data/", help="path to dataset")
parser.add_argument('--dataset_name', type=str, default="DimonGen/", help="path to dataset")
parser.add_argument('--model_path', type=str, default="checkpoints/", help="the path to saved models")
parser.add_argument('--output_dir', type=str, default="output/", help="the path to the output texts")

# list the method from: ["top_k", "top_p", "typical", "moe", "kgmoe", "moree"]
parser.add_argument('--method_name', type=str, default="moree", help="the name of the baseline methods to run")

parser.add_argument('--pretrained_model', type=str, default="facebook/bart-base", help="the name of the generation model")
parser.add_argument('--do_train', type=bool, default=False, help="")
parser.add_argument('--do_eval', type=bool, default=True, help="")
parser.add_argument('--training_epochs', type=int, default=30, help="the number of training epochs")
parser.add_argument('--learning_rate', type=int, default=3e-5, help="the generated sentence number for evaluation")
parser.add_argument('--batch_size', type=int, default=64, help="training batch size")
parser.add_argument('--max_sentence_len', type=int, default=50, help="the maximum generated sentence length according to the dataset")
parser.add_argument('--return_sentence_num', type=int, default=3, help="the generated sentence number for evaluation")

# for sampling methods
parser.add_argument('--top_k', type=int, default=50, help="the k size of top_k sampling")
parser.add_argument('--top_p', type=float, default=0.95, help="the probability mass p of top_p sampling")
parser.add_argument('--typical_p', type=float, default=0.90, help="the probability mass p of typical decoding")

# for MoE methods
parser.add_argument('--expert_num', type=int, default=3, help="the number of experts (hidden variables) to use")
parser.add_argument('--prompt_len', type=int, default=5, help="the token length of hidden varibles")

# for MoKGE methods
parser.add_argument('--training_gnn_epochs', type=int, default=5, help="the number of training epochs of gnn model")

# for moree methods
parser.add_argument('--retrieval_path', type=str, default="retrieval/output/", help="the path to external retrieval sentences")
# list the method from: ["baseline", "random", "em"], used for ablation study; baseline is for moree-irs, random is for moree-gen
parser.add_argument('--matching_method', type=str, default="em", help="the method to matching retrieval sentence to target generation")
parser.add_argument("--num_sent", type=int, default=3)

args = parser.parse_args()


if __name__ == "__main__":
    set_seed(0)

    if args.method_name not in ["top_k", "top_p", "typical", "moe", "kgmoe", "moree"]:
        raise ValueError(f"method_name:({args.method_name}) is not correct, please choose from ['top_k', 'top_p', 'typical', 'moe', 'kgmoe', 'moree'].")

    model = Trainer(args)

    if args.do_train:
        model.train_model()
    if args.do_eval:
        model.predict_result()

