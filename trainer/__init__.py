import transformers
import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelWithLMHead
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils.eval_acc_div import eval_accuracy_diversity, eval_top1_acc
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from utils.gnn import *
from nlgeval import compute_individual_metrics
import random


class Trainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.datasets = self.construct_dataset()

        # "top_k", "top_p", "typical" methods are using the same training strategy
        # expert_prompt is the hidden embedding for different experts in moe models.
        if args.method_name in ("top_k", "top_p", "typical"):
            self.model_path = args.model_path + "sampling"  + "/"
        elif args.method_name == "moree":
            self.model_path = args.model_path + args.method_name + "-" + args.matching_method + "/"
        else:
            self.model_path = args.model_path + args.method_name + "/"
        
        if args.do_train:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model) 
            if args.method_name in ("moe", "kgmoe", "moree"):
                self.expert_prompt = torch.randint(low=1, high=len(self.tokenizer), size=(args.expert_num, args.prompt_len))
        else:
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path + 'checkpoint-best/')
                # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path + 'checkpoint-20940/')
                if args.method_name in ("moe", "kgmoe", "moree"):
                    self.expert_prompt = torch.load(self.model_path + "expert_prompt.bin")
            except:
                raise NotImplementedError(f"Connot find pre-trained ({args.method_name}) model. You need to set args.do_train=True to train the model first!")

        if args.method_name == "moree":
            self.args.batch_size = int(self.args.batch_size / 2)

        trainer_args = Seq2SeqTrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=int(args.batch_size/args.return_sentence_num),
            save_total_limit=2,
            num_train_epochs=args.training_epochs,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="top1_bleu_4"
        )
        self.trainer = Seq2SeqTrainer(
            self.model,
            trainer_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["dev"],
            data_collator=self.DataCollator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )   

        if args.method_name == "kgmoe":
            self.gnn_model = GNN(self.tokenizer.vocab_size)
            gnn_model_path = args.model_path + "gnn/"
            if args.do_train or not os.path.exists(gnn_model_path):
                if not os.path.exists(gnn_model_path):
                    os.makedirs(gnn_model_path) 
                cp_train = [{k: v for k,v in i["input_ids"][1].items()} for i in self.datasets["train"]]
                cp_dev = [{k: v for k,v in i["input_ids"][1].items()} for i in self.datasets["dev"]]
                self.gnn_model = train_gnn_model(self.gnn_model, gnn_model_path + "model.ckpt", cp_train, cp_dev, args.training_gnn_epochs, args.batch_size)
            else:
                state_dict = torch.load(gnn_model_path + "model.ckpt")
                own_state = self.gnn_model.state_dict()
                for name, param in state_dict.items():
                    own_state[name].copy_(param)


    # training dataset is constructed in one-one generation
    # while validation and test datasets are one-many generation.
    def construct_dataset(self):
        data_path = self.args.data_path + self.args.dataset_name
        tokenizer = self.tokenizer
        data_dic = {}

        dataset_types = ["train", "dev", "test"]
        if self.args.method_name == "kgmoe":
            cp2id, id2cp, cpnet, cp_vocab = load_cpnet(self.args.data_path)
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
            for TYPE in (dataset_types):
                with open(data_path + TYPE + ".json", "r") as f:
                    lines = [json.loads(line) for line in f.readlines()]

                examples = []
                for line in tqdm(lines):
                    inp = match_concepts(line["inputs"], nlp, cp_vocab, cp2id)
                    lab = match_concepts(line["labels"], nlp, cp_vocab, cp2id)
                    nodes, edges, query, labels = construct_neighbor_graph(cpnet, inp, lab)
                    nodes = [self.tokenizer.encode(id2cp[n], add_special_tokens=False)[0] for n in nodes]
                    if TYPE == "train":
                        line_inputs = [tokenizer.encode(" ".join(line["inputs"]), max_length=self.args.max_sentence_len) for i in line["labels"]]
                        line_labels = [tokenizer.encode(i, max_length=self.args.max_sentence_len) for i in line["labels"]]
                    else:
                        line_inputs = [tokenizer.encode(" ".join(line["inputs"]))]
                        line_labels = [tokenizer.encode("\t".join(line["labels"]), max_length=self.args.max_sentence_len * self.args.return_sentence_num)]
                    # the second item in "input_ids" list is the inputs for kg embedding.
                    concepts = [tokenizer.encode(id2cp[n], add_special_tokens=False)[0] for n in nodes]
                    examples += [{
                                    "input_ids": [
                                        inp, 
                                        {
                                            "concepts": concepts,
                                            "edges": edges,
                                            "queries": query,
                                            "cp_labels": labels
                                        }
                                    ], 
                                    "labels": lab,
                                } for inp, lab in zip(line_inputs, line_labels)]
                data_dic[TYPE] = examples
                
        elif self.args.method_name == "moree":
            for TYPE in (dataset_types):
                with open(data_path + TYPE + ".json", "r") as f:
                    lines = [json.loads(line) for line in f.readlines()]
                with open(self.args.retrieval_path + TYPE + "_stage2.json", "r") as f:
                    re_lines = json.load(f)

                te_case = []
                examples = []
                for line, re in tqdm(zip(lines, re_lines), total=len(lines)):
                    re_sents = ["\t".join(sent) for sent in re]
                    if TYPE == "train":
                        line_inputs = [tokenizer.encode(" ".join(line["inputs"]), max_length=self.args.max_sentence_len * self.args.num_sent) for i in line["labels"]]
                        line_labels = [tokenizer.encode(i, max_length=self.args.max_sentence_len) for i in line["labels"]]
                    else:
                        line_inputs = [tokenizer.encode(" ".join(line["inputs"]))]
                        line_labels = [tokenizer.encode("\t".join(line["labels"]), max_length=self.args.max_sentence_len * self.args.return_sentence_num)]
                    # the second item in "input_ids" list is the inputs for retreival sentences embedding.
                    retrieval = [tokenizer.encode(sent, add_special_tokens=False, max_length=self.args.max_sentence_len) for sent in re_sents]
                    examples += [{
                                    "input_ids": [
                                        inp, 
                                        retrieval
                                    ], 
                                    "labels": lab,
                                } for inp, lab in zip(line_inputs, line_labels)]

                #     te_case.append([line["inputs"], line["labels"], re_sents])
                # with open("test_case.json", "w") as f:
                #     json.dump(te_case, f, indent=2)
                
                data_dic[TYPE] = examples
        else:
            for TYPE in (dataset_types):
                with open(data_path + TYPE + ".json", "r") as f:
                    lines = [json.loads(line) for line in f.readlines()]

                examples = []
                for line in tqdm(lines):
                    if TYPE == "train":
                        line_inputs = [tokenizer.encode(" ".join(line["inputs"]), max_length=self.args.max_sentence_len) for i in line["labels"]]
                        line_labels = [tokenizer.encode(i, max_length=self.args.max_sentence_len) for i in line["labels"]]
                    else:
                        line_inputs = [tokenizer.encode(" ".join(line["inputs"]))]
                        line_labels = [tokenizer.encode("\t".join(line["labels"]), max_length=self.args.max_sentence_len * self.args.return_sentence_num)]
                    examples += [{"input_ids": inp, "labels": lab} for inp, lab in zip(line_inputs, line_labels)]
                data_dic[TYPE] = examples

        return data_dic


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        tokenizer = self.tokenizer
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [label.split('\t') for label in decoded_labels]
        
        paired_inputs = []
        paired_labels = []
        for i in range(len(decoded_preds)):
            paired_inputs.append(decoded_preds[i:i + self.args.return_sentence_num])
            paired_labels.append(decoded_labels[i])
        result = eval_top1_acc(paired_inputs, paired_labels)
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: v for k, v in result.items()}


    def DataCollator(self, features, return_ts='pt'):
        '''
        Data Collator used for train, val, and test,
        code adapted from transformers.DataCollatorForSeq2Seq
        '''
        if self.args.method_name == "kgmoe":
            cp_features = [{k: v for k,v in i["input_ids"][1].items()} for i in features]
            features = [{"input_ids": i["input_ids"][0], "labels": i["labels"]} for i in features]

        if self.args.method_name == "moree":
            re_features = [j for i in features for j in i["input_ids"][1]]
            features = [{"input_ids": i["input_ids"][0], "labels": i["labels"]} for i in features]

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"])
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(features, padding=True, return_tensors=return_ts)
        
        if labels is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        if self.args.method_name == "kgmoe":
            concepts, node_labels, heads, tails, edge_labels, queries = kg_collate_fn(cp_features)
            node_repr = self.gnn_model.encode(concepts, heads, tails, edge_labels)
            outputs = self.gnn_model.decode(node_repr, queries)
            concept_ids = concepts.gather(dim=-1, index=outputs.topk(k=10, dim=-1).indices)
            masks_ids = torch.ones(concept_ids.shape)
            features["input_ids"] = torch.cat([features["input_ids"], concept_ids], dim=-1)
            features["attention_mask"] = torch.cat([features["attention_mask"], masks_ids], dim=-1)
        
        if self.args.method_name in ("moe", "kgmoe", "moree"):
            is_train = (labels is not None and '\t' not in self.tokenizer.decode(labels[0])) # val and test set labels are sentences concat with '\t'
            if self.args.method_name == "moree":
                re_features = self.tokenizer.pad({"input_ids": re_features}, padding=True, return_tensors=return_ts, max_length=self.args.max_sentence_len * self.args.num_sent)["input_ids"]
                features = self.construct_moe_dataset(features, is_train, re_features)
            else:
                features = self.construct_moe_dataset(features, is_train)
        return features

    
    # for moe model
    def construct_moe_dataset(self, batch_inputs, train=False, re_features=None):
        '''
        construct dataset with hidden variables of MOE.
        if train, the best hidden variable will be chosen to each input with hard EM algorithm
        if not train, simple concatenate all the hidden variables to the input.
        '''

        batch_size, label_len = batch_inputs['labels'].shape

        # construct prompt concatenated inputs
        mixture_ids_prompt = self.expert_prompt.repeat(batch_inputs['input_ids'].shape[0], 1)
        mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1)
        mixture_inputs = {k: self.repeat(v, self.args.expert_num) for k, v in batch_inputs.items()}

        
        mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
        mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)

        if re_features != None:   # for moree models
            assert len(re_features) == batch_size * self.args.expert_num    # you should set the expert num of generative model the same as retrieval model
            re_att = torch.full(re_features.shape, 1)

            if self.args.matching_method == "baseline":
                re_features = self.repeat(re_features[torch.arange(0, len(mixture_inputs['input_ids']), self.args.expert_num)], self.args.expert_num)
                mixture_inputs['input_ids'] = torch.cat([mixture_inputs['input_ids'], re_features], dim=1)
                mixture_inputs['attention_mask'] = torch.cat([mixture_inputs['attention_mask'], re_att], dim=1)
            elif self.args.matching_method == "random" or self.args.matching_method == "em":
                mixture_inputs['input_ids'] = torch.cat([mixture_inputs['input_ids'], re_features], dim=1)
                mixture_inputs['attention_mask'] = torch.cat([mixture_inputs['attention_mask'], re_att], dim=1)


        if train:
            # choose experts by model best output (only for train)
            self.model.eval()
            _inputs = mixture_inputs.copy()
            _inputs = {k: v.to(self.device) for k, v in _inputs.items()}
            labels = _inputs.pop("labels")
            model = self.model.to(self.device)
            outputs = model(**_inputs, use_cache=False)
            logits = outputs[0]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)).reshape(batch_size, self.args.expert_num, label_len)
            pad_mask = (batch_inputs['labels']==self.tokenizer.pad_token_id).view(batch_size, 1, label_len).to(self.device)
            mixture_ids = loss.masked_fill(pad_mask, 0).sum(dim=2).argmin(dim=1).unsqueeze(dim=1).type(torch.int64).cpu().detach()

            batch_inputs_new = batch_inputs
            expanded_mixture_ids = mixture_ids.expand(batch_size, self.args.prompt_len).unsqueeze(dim=1)
            input_ids_prompt = torch.gather(mixture_ids_prompt.view(batch_size, self.args.expert_num, -1), dim=1, index=expanded_mixture_ids).squeeze()
            attention_prompt = torch.full(input_ids_prompt.shape, 1)
            batch_inputs_new['input_ids'] = torch.cat([input_ids_prompt, batch_inputs_new['input_ids']], dim=1)
            batch_inputs_new['attention_mask'] = torch.cat([attention_prompt, batch_inputs_new['attention_mask']], dim=1)

            if re_features != None:
                expanded_re_ids = mixture_ids.expand(batch_size, re_features.shape[1]).unsqueeze(dim=1)
                if self.args.matching_method == "random":
                    expanded_re_ids = torch.randint(0, self.args.expert_num, expanded_re_ids.shape)
                input_re_prompt = torch.gather(re_features.view(batch_size, self.args.expert_num, -1), dim=1, index=expanded_re_ids).squeeze()
                re_att_prompt = torch.full(input_re_prompt.shape, 1)
                batch_inputs_new['input_ids'] = torch.cat([batch_inputs_new['input_ids'], input_re_prompt], dim=1)
                batch_inputs_new['attention_mask'] = torch.cat([batch_inputs_new['attention_mask'], re_att_prompt], dim=1)
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


    def train_model(self):
        self.trainer.train()
        self.trainer.save_model(self.model_path + 'checkpoint-best/')
        if self.args.method_name in ("moe", "kgmoe", "moree"):
            torch.save(self.expert_prompt, self.model_path + "expert_prompt.bin")
    

    def do_generation(self, batch, model):
        model = model.to(self.device)
        input_ids = torch.as_tensor(batch["input_ids"]).to(self.device)
        masks = torch.as_tensor(batch["attention_mask"]).to(self.device)

        if self.args.method_name == "top_k":
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=masks,
                do_sample=True, 
                max_length=self.args.max_sentence_len, 
                top_k=self.args.top_k,
                num_beams=self.args.return_sentence_num,
                # num_beams=1,
                num_return_sequences=self.args.return_sentence_num,
                return_dict_in_generate=True, 
                output_scores=True,
            )
        elif self.args.method_name == "top_p":
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=masks,
                do_sample=True, 
                max_length=self.args.max_sentence_len, 
                top_k=0,
                top_p=self.args.top_p,
                num_beams=self.args.return_sentence_num,
                num_return_sequences=self.args.return_sentence_num,
                return_dict_in_generate=True, 
                output_scores=True,
            )
        elif self.args.method_name == "typical":
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=masks,
                do_sample=True, 
                max_length=self.args.max_sentence_len, 
                top_k=0,
                typical_p=self.args.typical_p,
                num_beams=self.args.return_sentence_num,
                num_return_sequences=self.args.return_sentence_num,
                return_dict_in_generate=True, 
                output_scores=True,
            )
        else:
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=masks,
                max_length=self.args.max_sentence_len, 
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True, 
                output_scores=True,
            )

        generated_ids = outputs.sequences.detach().cpu()
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logits = torch.stack(outputs.scores, 0).transpose(0, 1).softmax(-1).detach().cpu()
        return predictions, logits


    def write_results(self, output_dir, predictions, labels):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_pred = output_dir + '/predictions.json'
        out_ref = output_dir + '/targets.json'

        with open(out_pred, 'w') as eval_out:
            for pred in predictions:
                eval_out.write(json.dumps(pred) + '\n')

        targets = [i.split("\t") for i in labels]
        with open(out_ref, 'w') as eval_truth:
            for gt in targets:
                eval_truth.write(json.dumps(gt) + '\n')

        metrics = eval_accuracy_diversity(out_pred, out_ref, self.args.data_path + self.args.dataset_name)
        with open(output_dir +  "/metrics.json", 'w') as metric_out:
            json.dump(metrics, metric_out, indent=1)
        print(metrics)


    def predict_result(self, gt_dataset=None):
        if gt_dataset == None:
            gt_dataset = self.datasets["test"] 
        dataloader = self.trainer.get_test_dataloader(gt_dataset)

        if self.args.method_name in ("kgmoe", "moree"):
            sources = [i.split(" ") for i in self.tokenizer.batch_decode([j["input_ids"][0] for j in gt_dataset], skip_special_tokens=True)]
        else:
            sources = [i.split(" ") for i in self.tokenizer.batch_decode([j["input_ids"] for j in gt_dataset], skip_special_tokens=True)]
        labels = self.tokenizer.batch_decode([j["labels"] for j in gt_dataset], skip_special_tokens=True)

        self.model.eval()
        preds = []
        for batch in tqdm(dataloader):
            pred, _ = self.do_generation(batch, self.model)
            preds += pred
        predictions = [preds[i:i+self.args.return_sentence_num] for i in range(0, len(preds), self.args.return_sentence_num)]
        assert len(sources) == len(predictions)

        output_dir = self.args.output_dir + self.args.method_name
        if self.args.method_name == "moree":
            output_dir = output_dir + "_" + self.args.matching_method
        self.write_results(output_dir, predictions, labels)
    