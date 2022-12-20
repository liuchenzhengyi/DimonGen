import os
import numpy as np
from collections import defaultdict
import json

from nlgeval import compute_metrics
from nlgeval import compute_individual_metrics


def get_all_files(path):

    if os.path.isfile(path): return [path]

    return [f for d in os.listdir(path)
              for f in get_all_files(os.path.join(path, d))]
    

def eval_top1_acc(predictions, targets):
    # hyp_top1 and refs are both a list of string
    # E.g., [hyp1 for data sample 1, hyp2 for ...]
    if isinstance(predictions[0], str):
        hyps_top1 = predictions
    else:
        hyps_top1 = [pred[0] for pred in predictions]
    top1_metrics = compute_metrics(hyp_list=hyps_top1, ref_list=targets)
    top1_metrics = {f'top1_{k}': v for k, v in top1_metrics.items()}
    return top1_metrics


def eval_topk_acc(predictions, targets):
    hyps_best = []
    for hyp, ref in zip(predictions, targets):
        hyp_score_list = [compute_individual_metrics(ref, h)['bleu_4'] for h in hyp]
        hyps_best.append(hyp[np.argmax(hyp_score_list)])
    
    topk_metrics = compute_metrics(ref_list=targets, hyp_list=hyps_best)
    topk_metrics = {f'topk_{k}': v for k, v in topk_metrics.items()}
    return topk_metrics


# # choose top one hypothesis
# def eval_top1_acc(predictions, targets):
#     hyps_best = []
#     for hyp, ref in zip(predictions, targets):
#         hyp_score_list = [compute_individual_metrics(ref, h)['bleu_4'] for h in hyp]
#         hyps_best.append(hyp[np.argmax(hyp_score_list)])
    
#     topk_metrics = compute_metrics(ref_list=targets, hyp_list=hyps_best)
#     topk_metrics = {f'top1_{k}': v for k, v in topk_metrics.items()}
#     return topk_metrics


# def eval_topk_acc(predictions, targets):
#     hyp_list = []
#     ref_list = []
#     for hyp, ref in zip(predictions, targets):
#         hyp_list += [h for h in hyp]
#         ref_list += [ref for _ in range(len(hyp))]
#     topk_metrics = compute_metrics(ref_list=ref_list, hyp_list=hyp_list)
#     topk_metrics = {f'topk_{k}': v for k, v in topk_metrics.items()}
#     return topk_metrics


# # choose top one reference
# def eval_top1_acc(predictions, targets):
#     hyp_list = []
#     ref_list = []
#     for hyp, ref in zip(predictions, targets):
#         hyp_list += [h for h in hyp]
#         ref_list += [ref for _ in range(len(hyp))]
#     topk_metrics = compute_metrics(ref_list=ref_list, hyp_list=hyp_list)
#     topk_metrics = {f'top1_{k}': v for k, v in topk_metrics.items()}
#     return topk_metrics


# def eval_topk_acc(predictions, targets):
#     hyp_score_list = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": [], "rouge_l": []}
#     for hyp, ref in zip(predictions, targets):
#         for h in hyp:
#             h_score_list = [compute_individual_metrics(r, h) for r in ref]
#             for k in hyp_score_list:
#                 hyp_score_list[k].append(max([score[k] for score in h_score_list]))
        
#     topk_metrics = {}
#     for k in hyp_score_list:
#         topk_metrics[k] = sum(hyp_score_list[k]) / len(hyp_score_list[k])
#     topk_metrics = {f'topk_{k}': v for k, v in topk_metrics.items()}
#     return topk_metrics


def eval_self_bleu(predictions):
    hyp_list, ref_list = [], []
    for hyps in predictions:
        for i in range(len(hyps)):
            hyp_list.append(hyps[i])
            ref_list.append(hyps[:i] + hyps[i+1:])
    self_metrics = compute_metrics(hyp_list=hyp_list, ref_list=ref_list)
    self_metrics = {f'self_{k}': v for k, v in self_metrics.items()}
    return self_metrics


def eval_entropy_distinct(predictions):
    diversity_metrics = {}
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for pred in predictions:
        for gg in pred:
            g = gg.rstrip('2').split()
            for n in range(4):
                for idx in range(len(g)-n):
                    ngram = ' '.join(g[idx:idx+n+1])
                    counter[n][ngram] += 1
        
    for n in range(4):
        entropy_score = 0
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            entropy_score += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        diversity_metrics[f'entropy_{n+1}'] = entropy_score

    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        diversity_metrics[f'distinct_{n+1}'] = (len(counter[n].values())+0.0) / total

    return diversity_metrics


def eval_successful_rate(inputs, predictions):
    successful_metrics = {}
    score_list = []
    for inp, pred in zip(inputs, predictions):
        s = 0
        for i in inp:
            for j in pred:
                if i in j:
                    s += 1
        score_list.append(s / (len(inp) * len(pred)))

    successful_metrics["successful_rate"] = sum(score_list) / len(score_list)
    return successful_metrics


def eval_accuracy_diversity(hyp_path, ref_path, dataset=None):
    metrics = {}

    with open(hyp_path, "r") as f:
        predictions = [json.loads(line) for line in f]
    with open(ref_path, "r") as f:
        targets = [json.loads(line) for line in f]

    metrics.update(eval_top1_acc(predictions, targets))
    metrics.update(eval_topk_acc(predictions, targets))
    metrics.update(eval_self_bleu(predictions))
    metrics.update(eval_entropy_distinct(predictions))

    if dataset:
        with open(dataset + "test.json", "r") as f:
            inputs = [json.loads(line)["inputs"] for line in f.readlines()]

        metrics.update(eval_successful_rate(inputs, predictions))
        
    return metrics
