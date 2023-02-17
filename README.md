# DimonGen: Diversified Generative Commonsense Reasoning for Explaining Concept Relationships

## MoREE: Mixture of Retrieval-Enhanced Experts

![image](https://user-images.githubusercontent.com/47152740/219499937-abdaf47e-443c-46f0-96ce-dedb01d3ee00.png)

## Data

Our extracted DimonGen dataset is vailable at here: [dataset](data/DimonGen).

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

## Citation

```
@article{liu2022dimongen,
  title={DimonGen: Diversified Generative Commonsense Reasoning for Explaining Concept Relationships},
  author={Liu, Chenzhengyi and Huang, Jie and Zhu, Kerui and Chang, Kevin Chen-Chuan},
  journal={arXiv preprint arXiv:2212.10545},
  year={2022}
}
```
