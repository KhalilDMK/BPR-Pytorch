# TopKBPR
Pytorch implementation of the paper "BPR: Bayesian Personalized Ranking from Implicit Feedback".

Link to the paper: https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf

## Environment settings
We use Pytorch 1.1.0.

## Description
This repository includes code to train the BPR model and tune its hyperparameters: Run "Train_BPR.py". You can tune the hyperparameters by updating the configuration dictionary. 

We made two evaluation procedures available (change "loo_eval" in the config dictionary):
* <b>Leave-One-Out evaluation:</b> where the last interaction of each user is left out as test data and 100 negative items are sampled for each user. The positive item is ranked with respect to the negative items in terms of Hit ratio at cutoff K (HR@K) and Normalized Discounted Cumulative Gain at cutoff K (NDCG@K).
* <b>Explicit evaluation:</b> random train/test split where the test items are ranked for each user in terms of Mean Average Precision at cutoff K (MAP@K) and Normalized Discounted Cumulative Gain at cutoff K (NDCG@K). The evaluation procedure uses the original explicit ratings to get the true rank of each item.

After training, the weights of the best model in terms of NDCG@K will be saved in an Output folder.

## Datasets
You can train the model on the Movielens 100K and Movielens 1M datasets (Change the "dataset_name" variable). 
