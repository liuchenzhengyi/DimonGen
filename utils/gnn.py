import networkx as nx
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_max, scatter_mean, scatter_add
import transformers


class GNN(torch.nn.Module):
    def __init__(self, vocab_size, padding_idx=0, embed_size=512, layer_number=2):
        super().__init__()
        self.embed_word = nn.Embedding(vocab_size, embed_size, padding_idx)
        self.layer_number = layer_number
        self.W_s = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.layer_number)]) 
        self.W_n = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.layer_number)])
        # cat the dim of query and the candidates together
        self.output_layer = nn.Sequential(nn.Linear(embed_size * 2, embed_size), nn.Tanh(), nn.Linear(embed_size, 1))


    def gcn_layer(self, concept_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        '''
        bsz = head.size(0)
        mem_t = head.size(1)
        mem = concept_hidden.size(1)
        hidden_size = concept_hidden.size(2)
        update_hidden = torch.zeros_like(concept_hidden).to(concept_hidden.device).float()
        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float()
        count_out = torch.zeros(bsz, mem).to(head.device).float()

        o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, tail, dim=1, out=update_hidden)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_hidden)
        scatter_add(count, head, dim=1, out=count_out)

        act = nn.ReLU()
        update_hidden = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_hidden) / count_out.clamp(min=1).unsqueeze(2)
        update_hidden = act(update_hidden)

        return update_hidden


    def encode(self, concept_ids, head, tail, triple_label):
        concept_hidden = self.embed_word(concept_ids)
        for i in range(self.layer_number):
            concept_hidden = self.gcn_layer(concept_hidden, head, tail, triple_label, i)
        return concept_hidden


    def decode(self, node_logits, query_index):
        query_logits = node_logits.gather(
            dim=1, 
            index=query_index.unsqueeze(2).expand(query_index.size(0), query_index.size(1), node_logits.size(2))
        ).mean(dim=1, keepdim=True)
        query_logits = query_logits.expand(node_logits.shape)
        outputs = self.output_layer(torch.cat([query_logits, node_logits], dim=-1)).squeeze(dim=-1)
        return outputs


    def compute_loss(self, outputs, node_labels):
        node_loss = F.binary_cross_entropy_with_logits(
            outputs.float(), 
            node_labels.float(), 
            reduction='none'
        )
        return node_loss.sum()


def load_cpnet(data_path):
    concept2id = {}
    id2concept = {}
    with open(data_path + "concept.txt", "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    cpnet = nx.read_gpickle(data_path + "cpnet_simple.graph")

    cpnet_vocab = set([id2concept[n] for n in cpnet])
    return concept2id, id2concept, cpnet, cpnet_vocab


def match_concepts(sentences, nlp, cp_vocab, cp2id):
    sent = " ".join(sentences).lower().strip()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cp_vocab:
            if t.pos_ == "NOUN" or t.pos_ == "VERB" or t.pos_ == "PROPN":
                res.add(cp2id[t.lemma_])

    return res


def construct_neighbor_graph(cpnet, in_nodes, out_nodes, R=2, max_N=70):
    start = in_nodes.copy()
    Vts = {}     # id->pos
    for x in start:
        Vts[x] = len(Vts)  
    for t in range(R):
        V = {}  # id->frequency
        for s in start:
            for n in cpnet[s]:
                if n not in Vts:    # BFS
                    if n not in V:
                        V[n] = 0
                    V[n] += 1 

        start = sorted(V, key=V.get)[:max_N]
        for x in start:
            Vts[x] = len(Vts) 
    
    node_ids = list(Vts.keys())
    edge_ids = list(cpnet.subgraph(node_ids).edges)
    edges = [[Vts[e[0]], Vts[e[1]]] for e in edge_ids]  # id->pos

    query = [Vts[i] for i in in_nodes if i in Vts]
    labels = [1 if i in out_nodes else 0 for i in node_ids]

    return node_ids, edges, query, labels


def kg_collate_fn(batch):
    max_seq_length = max(len(l["concepts"]) for l in batch)
    concepts = torch.tensor([l["concepts"] + [0] * (max_seq_length - len(l["concepts"])) for l in batch])
    node_labels = torch.tensor([l["cp_labels"] + [-1] * (max_seq_length - len(l["cp_labels"])) for l in batch])
    max_edge_length = max(len(l["edges"]) for l in batch)
    heads = torch.tensor([[e[0] for e in l["edges"]] + [0] * (max_edge_length - len(l["edges"])) for l in batch])
    tails = torch.tensor([[e[1] for e in l["edges"]] + [0] * (max_edge_length - len(l["edges"])) for l in batch])
    edge_labels = torch.tensor([[1] * len(l["edges"]) + [0] * (max_edge_length - len(l["edges"])) for l in batch])
    max_query_length = 2
    queries = torch.tensor([l["queries"] + [0] * (max_query_length - len(l["queries"])) for l in batch])
    return [concepts, node_labels, heads, tails, edge_labels, queries]


def train_gnn_model(gnn_model, save_path, train_set, dev_set, epoches=10, batch_size=64):
    print("***** Running training on GNN model *****")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=kg_collate_fn,
        pin_memory=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=kg_collate_fn,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(params=gnn_model.parameters(), lr=1e-4)
    scheduler = transformers.optimization.get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=1200,
        num_training_steps=epoches * len(train_set)
    )
    best_eval_score = float('inf')
    for epoch in range(epoches):
        for i in tqdm(train_loader):
            concepts, node_labels, heads, tails, edge_labels, queries = i
            node_repr = gnn_model.encode(concepts, heads, tails, edge_labels)
            outputs = gnn_model.decode(node_repr, queries)
            loss = gnn_model.compute_loss(outputs, node_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        eval_score = 0
        with torch.no_grad():
            for i in tqdm(dev_loader):
                concepts, node_labels, heads, tails, edge_labels, queries = i
                node_repr = gnn_model.encode(concepts, heads, tails, edge_labels)
                outputs = gnn_model.decode(node_repr, queries)
                loss = gnn_model.compute_loss(outputs, node_labels)
                eval_score += loss.item()
        if eval_score < best_eval_score:
            best_eval_score = eval_score
            torch.save(gnn_model.state_dict(), save_path)
    
    state_dict = torch.load(save_path)
    own_state = gnn_model.state_dict()
    for name, param in state_dict.items():
        own_state[name].copy_(param)
    
    print("GNN model checkpoints saved at" + save_path)
    print()
    return gnn_model

