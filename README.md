# DimonGen: Diversified Generative Commonsense Reasoning for Explaining Concept Relationships

## Data

Our extracted DimonGen dataset is vailable at here: [dataset](data/DimonGen).

## MoREE: Mixture of Retrieval-Enhanced Experts

![MoREE framwork](doc/model.pdf)

## Environment

Python 3.9.12, Pytorch 1.11.0, Hugging Face Transformers 4.19.4

## Training & Evaluation

download the external knowledge documents from [here](https://drive.google.com/file/d/1okinbYNp0D66m2Oj-PCmCJGo3Bnzfq36/view?usp=share_link), and put the files into data folder, and then run
```
python retrieval/initial_retriever.py
python retrieval/mixture_of_scorer.py
```

Set the default --method_name moree in main.py then run
```
python main.py
```

To run baseline methods, you need to set the method_name to different baseline methods.
