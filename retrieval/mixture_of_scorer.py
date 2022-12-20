import argparse
from cProfile import label
from platform import java_ver
from tracemalloc import is_tracing
from turtle import Turtle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import json
from tqdm import tqdm
import random
import os
import numpy as np
import sklearn
import transformers


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MoREETrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        expert_num = 1,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, \
            model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

        self.mixture_ids = None # indicate expert idx of inputs in each batch
        self.alpha = 100
        self.beta = 100
        self.expert_num = expert_num


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(logits, labels)
        if self.mixture_ids != None:
            expert_probs = []
            soft_logits = torch.softmax(logits, axis=1)
            for val in torch.unique(self.mixture_ids):
                expert_probs.append(torch.mean(soft_logits[(self.mixture_ids == val).squeeze()], axis=0))
            regulariztion_prob = self.jenson_shannon_divergence(expert_probs)
            # regulariztion_prob = 0
            loss = loss + self.alpha * regulariztion_prob 

            # expert_num = []
            # for val in range(self.expert_num):
            #     expert_num.append(torch.sum(self.mixture_ids == val).detach().cpu())
            # kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            # expert_num_tensor = torch.stack(expert_num) 
            # expert_num_tensor = expert_num_tensor / torch.sum(expert_num_tensor)
            # uni_num = torch.ones(expert_num_tensor.shape) / expert_num_tensor.shape[0]
            # regulariztion_num = kl_loss(torch.nn.functional.log_softmax(expert_num_tensor.float(), dim=-1), uni_num)
            # loss = loss + self.beta * regulariztion_num

            self.mixture_ids = None
            
            # print(expert_num_tensor)
            # for i in range(len(expert_num)):
            #     print("The number of expert " + str(i) + " is:", expert_num[i].tolist())

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def jenson_shannon_divergence(probs):
        # information radius KL Divergence form of multiple distribution
        loss = []
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        total_prob = torch.mean(torch.stack(probs), axis=0)
        for i in range(len(probs)):
            pred = probs[i]
            loss.append(kl_loss(torch.nn.functional.log_softmax(pred, dim=-1), total_prob))

        return torch.mean(torch.stack(loss))


class IR_Process():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if args.do_train:
            self.model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model) 
            self.expert_prompt = torch.randint(low=1, high=len(self.tokenizer), size=(args.expert_num, args.prompt_len))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.save_model_path + 'checkpoint-best/')
            self.expert_prompt = torch.load(self.args.save_model_path + "expert_prompt.bin")

        training_args = TrainingArguments(
            args.save_model_path,
            evaluation_strategy = "steps",
            save_strategy = "steps",
            learning_rate=2e-5,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            save_total_limit=2,
            num_train_epochs=args.training_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_score",
        )
        self.trainer = MoREETrainer(
            self.model,
            training_args,
            tokenizer=self.tokenizer,
            data_collator=self.DataCollator,
            compute_metrics=self.compute_metrics,
            expert_num=self.args.expert_num
        )


    def construct_dataset(self, org_data_path, re_data_path):
        tokenizer = self.tokenizer
        data_dic = {}

        dataset_types = ["train", "dev", "test"]
        for TYPE in (dataset_types):
            with open(org_data_path + self.args.dataset_name + TYPE + ".json", "r") as f:
                lines = [json.loads(line) for line in f.readlines()]
            with open(re_data_path + TYPE + "_stage1.json", "r") as f:
                re_lines = json.load(f)
            pool = [ex for line in re_lines for ex in line]

            examples = []
            for i in tqdm(range(len(lines))):
                inp_text = " ".join(lines[i]["inputs"])

                if self.is_training:
                    pos_num = len(lines[i]["labels"])
                    line_inputs = [tokenizer.encode(inp_text, j, max_length=self.args.max_length) for j in lines[i]["labels"]]
                    line_labels = [1 for j in range(pos_num)]
                
                    n = min(pos_num, len(re_lines[i]))
                    line_inputs += [tokenizer.encode(inp_text, j, max_length=self.args.max_length) for j in random.sample(re_lines[i], n)]
                    line_labels += [0 for j in range(n)]
                    if n < pos_num:
                        line_inputs += [tokenizer.encode(inp_text, j, max_length=self.args.max_length) for j in random.sample(pool, pos_num - n)]
                        line_labels += [0 for j in range(pos_num - n)]

                    if TYPE == "train":
                        examples += [{"input_ids": inp, "labels": lab} for inp, lab in zip(line_inputs, line_labels)]
                    else:
                        examples += [{"input_ids": inp, "labels": [lab]} for inp, lab in zip(line_inputs, line_labels)]
                else:
                    line_inputs = [tokenizer.encode(inp_text, j, max_length=self.args.max_length) for j in re_lines[i]]
                    line_sent = re_lines[i]
                    examples += [{"input_ids": line_inputs[j], "idx": i, "sentences": line_sent[j]} for j in range(len((line_inputs)))]
                
            data_dic[TYPE] = examples

        return data_dic


    def DataCollator(self, features, return_tensors="pt"):
        '''
        Data Collator used for train, val, and test,
        code adapted from transformers.DataCollatorWithPadding
        '''
        is_train = False
        if "labels" in features[0]:
            if isinstance(features[0]["labels"], int):
                is_train = True
            else:   # tell the data collator to load different type of expert data.
                for i in range(len(features)):
                    features[i]["labels"] = features[i]["labels"][0]

        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.args.max_length,
            return_tensors=return_tensors,
        )
        
        features = self.construct_moe_dataset(features, is_train)

        return features


    def construct_moe_dataset(self, batch_inputs, is_train):
        '''
        construct dataset with hidden variables of MOE.
        the best hidden variable will be chosen to each input with hard EM algorithm
        '''
        # construct prompt concatenated inputs
        mixture_ids_prompt = self.expert_prompt.repeat(batch_inputs['input_ids'].shape[0], 1)
        mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1)
        mixture_inputs = {k: self.repeat(v, self.args.expert_num) for k, v in batch_inputs.items()}
        mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
        mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)

        if is_train:
            self.model.eval()
            _inputs = mixture_inputs.copy()
            _inputs = {k: v.to(self.device) for k, v in _inputs.items()}
            labels = _inputs.pop("labels")
            model = self.model.to(self.device)
            outputs = model(**_inputs)
            logits = outputs[0]

            batch_size = len(batch_inputs['labels'])
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels).reshape(batch_size, self.args.expert_num)
            loss = loss + torch.normal(0, 0.1, loss.shape).to(self.device)  # noise term
            mixture_ids = loss.argmin(dim=1).unsqueeze(dim=1).type(torch.int64).cpu().detach()
            expanded_mixture_ids = mixture_ids.expand(batch_size, self.args.prompt_len).unsqueeze(dim=1)
            input_ids_prompt = torch.gather(mixture_ids_prompt.view(batch_size, self.args.expert_num, -1), dim=1, index=expanded_mixture_ids).squeeze()
            attention_prompt = torch.full(input_ids_prompt.shape, 1)

            batch_inputs_new = batch_inputs
            batch_inputs_new['input_ids'] = torch.cat([input_ids_prompt, batch_inputs['input_ids']], dim=1)
            batch_inputs_new['attention_mask'] = torch.cat([attention_prompt, batch_inputs['attention_mask']], dim=1)

            self.trainer.mixture_ids = mixture_ids
        else:
            batch_inputs_new = mixture_inputs

        return batch_inputs_new


    @staticmethod
    def repeat(tensor, K):
        # [B, ...] => [B*K, ...] Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
        if isinstance(tensor, torch.Tensor):
            B, *size = tensor.size()
            expand_size = B, K, *size
            tensor = tensor.unsqueeze(1).expand(*expand_size).contiguous().view(B * K, *size)
            return tensor
        elif isinstance(tensor, list):
            out = []
            for x in tensor:
                for _ in range(K):
                    out.append(x.copy())
            return out


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        predictions = np.argmax(predictions, axis=1)
        f1 = sklearn.metrics.f1_score(labels, predictions)
        return {"f1_score": f1}

    
    def train_model(self):
        self.is_training = True

        dataset = self.construct_dataset(args.org_data_path, args.re_data_path)
        self.trainer.train_dataset = dataset["train"]
        self.trainer.eval_dataset = dataset["dev"]

        self.trainer.train()
        self.trainer.save_model(self.args.save_model_path + 'checkpoint-best/')
        torch.save(self.expert_prompt, self.args.save_model_path + "expert_prompt.bin")
        acc = self.trainer.predict(dataset["test"])
        print("The accuracy score for is: ", acc.metrics)

    
    def retrieve_sentences(self):
        self.is_training = False

        data = self.construct_dataset(self.args.org_data_path, self.args.re_data_path)

        for TYPE in ("dev", "test", "train"):
            dataset = data[TYPE]

            scores = []
            dataloader = self.trainer.get_test_dataloader(dataset)
            torch.cuda.empty_cache()
            self.model.eval()
            self.model = self.model.to(self.device)
            for batch in tqdm(dataloader):
                _inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**_inputs)[0]
                batch_score = outputs.softmax(dim=-1)[:, 1]
                scores += batch_score.detach().cpu()

            assert len(scores) == len(dataset) * self.args.expert_num
            
            with open(self.args.org_data_path + self.args.dataset_name + TYPE + ".json", "r") as f:
                length = len(f.readlines())

            re_scores = [[[] for i in range(length)] for j in range(self.args.expert_num)]
            re_sents = [[] for i in range(length)]
            for i, ex in enumerate(dataset):
                for j in range(len(re_scores)):
                    re_scores[j][ex["idx"]].append(scores[i * self.args.expert_num + j])
                re_sents[ex["idx"]].append(ex["sentences"])

            res = [[[] for j in range(self.args.expert_num)] for i in range(length)]
            for i in range(length):
                for j in range(self.args.expert_num):
                    sorted_sent = [x for _, x in sorted(zip(re_scores[j][i], re_sents[i]), key=lambda pair: -pair[0])]
                    res[i][j] = sorted_sent[:self.args.num_sent]
                    while len(res[i][j]) < self.args.num_sent:
                        res[i][j].append("")

            # for i in range(length):
            #     temp_sents = []
            #     temp_pointers = []
            #     for j in range(self.args.expert_num):
            #         temp_sents.append([x for _, x in sorted(zip(re_scores[j][i], re_sents[i]), key=lambda pair: -pair[0])])
            #         temp_pointers.append(0)
            #     used_sents = []
            #     for k in range(self.args.num_sent):
            #         for j in range(self.args.expert_num):
            #             while temp_pointers[j] < len(temp_sents[j]) and temp_sents[j][temp_pointers[j]] in used_sents:
            #                 temp_pointers[j] = temp_pointers[j] + 1
            #             if temp_pointers[j] < len(temp_sents[j]):
            #                 res[i][j].append(temp_sents[j][temp_pointers[j]])
            #                 used_sents.append(temp_sents[j][temp_pointers[j]])
            #                 temp_pointers[j] = temp_pointers[j] + 1
            #             else:
            #                 res[i][j].append("")

            output_dir = self.args.re_data_path
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_dir + TYPE + "_stage2.json", "w") as f:
                json.dump(res, f, indent=2)
    

    def test_model(self):
        self.model = self.model.to(torch.device("cpu"))
        # inputs = self.tokenizer("cloud mountain", "An object stands in front of a structure before a drawing of mountains and clouds against the wall.", return_tensors="pt")
        inputs = self.tokenizer("cloud mountain", "clouds and fog in the mountains", return_tensors="pt")
        
        mixture_ids_prompt = self.expert_prompt.repeat(inputs['input_ids'].shape[0], 1)
        mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1)
        mixture_inputs = {k: self.repeat(v, self.args.expert_num) for k, v in inputs.items()}
        mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
        mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)

        outputs = self.model(**mixture_inputs)
        print(torch.argmax(outputs[0], axis=1))
        exit()


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument('--do_train', type=bool, default=True, help="")
    parser.add_argument("--pretrained_model", type=str, default="roberta-base")
    parser.add_argument("--org_data_path", type=str, default="data/")
    parser.add_argument('--dataset_name', type=str, default="DimonGen/", help="path to dataset")
    parser.add_argument("--re_data_path", type=str, default="retrieval/output/")
    parser.add_argument("--save_model_path", type=str, default="checkpoints/retrieval/")

    parser.add_argument("--expert_num", type=int, default=3)
    parser.add_argument('--prompt_len', type=int, default=5, help="the token length of hidden varibles")

    parser.add_argument("--num_sent", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--training_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")

    # used for ablation study
    # parser.add_argument("--use_reg", type=bool, default=False, help="to used the regulazition form")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    transformers.set_seed(args.seed)

    model = IR_Process(args)

    # model.test_model()

    if args.do_train:
        model.train_model()
    model.retrieve_sentences()
