import pandas as pd
import numpy as np
from Code.BPR_model import BPREngine
from Code.data import SampleGenerator

# Load Movielens 1M Data
#ml1m_dir = '../Data/ml-1m/ratings.dat'
#dataset_name = 'ml-1m'
#ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

# Load Movielens 100K Data
ml1m_dir = '../Data/ml-100k/u.data'
dataset_name = 'ml-100k'
ml1m_rating = pd.read_csv(ml1m_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

# Reindex data
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

# Define hyperparameters
BPR_config = {'alias': 'collab_conv_' + dataset_name,
              'num_epoch': 10,
              'batch_size': 500,
              'lr': 0.001,
              #'optimizer': 'sgd',
              #'sgd_momentum': 0.9,
              #'optimizer': 'rmsprop',
              #'rmsprop_alpha': 0.99,
              #'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'num_users': len(ml1m_rating['userId'].unique()),
              'num_items': len(ml1m_rating['itemId'].unique()),
              'test_rate': 0.2,  # Test rate for random train/test split. Used when 'loo_eval' is set to False.
              'num_latent': 50,
              'weight_decay': 0,
              'l2_regularization': 0,
              'use_cuda': True,
              'device_id': 0,
              'top_k': 10,  # k in MAP@k, HR@k and NDCG@k.
              'loo_eval': True,  # True: LOO evaluation with HR@k and NDCG@k. False: Random train/test split
              # evaluation with MAP@k and NDCG@k.
              'model_dir_explicit':'../Output/checkpoints/{}_Epoch{}_MAP@{}_{:.4f}_NDCG@{}_{:.4f}.model',
              'model_dir_implicit':'../Output/checkpoints/{}_Epoch{}_NDCG@{}_{:.4f}_HR@{}_{:.4f}.model'}

config = BPR_config

# DataLoader
sample_generator = SampleGenerator(ml1m_rating, config)
evaluation_data = sample_generator.test_data_loader(config['batch_size'])

# Specify the exact model
engine = BPREngine(config)

# Initialize list of optimal results
best_performance = [0] * 3

best_model = ''
for epoch in range(config['num_epoch']):
    print('Training epoch {}'.format(epoch))
    train_loader = sample_generator.train_data_loader(config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    if config['loo_eval'] == True:
        hr, ndcg = engine.evaluate(evaluation_data, epoch_id=epoch)
        print('-' * 80)
        best_model, best_performance = engine.save_implicit(config['alias'], epoch, ndcg, hr, config['num_epoch'], best_model, best_performance)
    else:
        map, ndcg = engine.evaluate(evaluation_data, epoch_id=epoch)
        print('-' * 80)
        best_model, best_performance = engine.save_explicit(config['alias'], epoch, map, ndcg, config['num_epoch'], best_model, best_performance)