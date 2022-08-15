import pickle as pk
import numpy as np
import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

folder = 'Amazon'

print('Read data...')
u_voc = pk.load(open('user_vocab.pkl', 'rb'))
i_voc = pk.load(open('item_vocab.pkl', 'rb'))
c_voc = pk.load(open('category_vocab.pkl', 'rb'))

train = pd.read_csv('train_data', sep='\t', names=['label', 'user', 'item', 'category', 'timestamp', 'seq_item', 'seq_cat', 'seq_time'])
valid = pd.read_csv('valid_data', sep='\t', names=['label', 'user', 'item', 'category', 'timestamp', 'seq_item', 'seq_cat', 'seq_time'])
test = pd.read_csv('test_data', sep='\t', names=['label', 'user', 'item', 'category', 'timestamp', 'seq_item', 'seq_cat', 'seq_time'])

train = train[['label', 'user', 'item', 'category', 'timestamp']]
valid = valid[['label', 'user', 'item', 'category', 'timestamp']]
test = test[['label', 'user', 'item', 'category', 'timestamp']]

print('done')

print('negative sampling')
samp = train[['item', 'category']].sample(frac=1).reset_index(drop=True)
neg_train = train.copy()
neg_train['item'] = samp['item']
neg_train['category'] = samp['category']
neg_train['label'] = 0

samp = train[['item', 'category']].sample(frac=1).reset_index(drop=True)
neg_train2 = train.copy()
neg_train2['item'] = samp['item']
neg_train2['category'] = samp['category']
neg_train2['label'] = 0

new_train = pd.concat( [train, neg_train, neg_train2], axis=0, ignore_index=True)

print('done')

print('generating data...')
# convert time
new_train['date'] = pd.to_datetime(new_train['timestamp'], unit='s').dt.strftime('%Y%m')
valid['date'] = pd.to_datetime(valid['timestamp'], unit='s').dt.strftime('%Y%m')
test['date'] = pd.to_datetime(test['timestamp'], unit='s').dt.strftime('%Y%m')

trans_col = ['user', 'item', 'category', 'date']
merge_pd = pd.concat( [new_train, valid, test], axis=0)
merge_pd['item_cat'] = merge_pd['item'] + '_' + merge_pd['category']
all_values = np.concatenate((merge_pd['user'].unique(), merge_pd['item'].unique(), merge_pd['category'].unique(), merge_pd['date'].unique()))
le.fit(all_values)

t_train = pd.DataFrame()
t_valid = pd.DataFrame()
t_test = pd.DataFrame()

t_train['rating'] = new_train['label']
t_valid['rating'] = valid['label']
t_test['rating'] = test['label']
for col in trans_col:
    t_train[col] = le.transform(new_train[col])
    t_valid[col] = le.transform(valid[col])
    t_test[col] = le.transform(test[col])

data_col = ['user', 'item', 'rating', 'date']
t_train[data_col].to_csv(f'{folder}/train_data.csv', index=0)
t_valid[data_col].to_csv(f'{folder}/valid_data.csv', index=0)
t_test[data_col].to_csv(f'{folder}/test_data.csv', index=0)

print('done')

print('create dict file')

feature_dict = {}
for i in range(len(le.classes_)):
    feature_dict[le.classes_[i]] = i

user_dict = {}
for user in merge_pd['user'].unique():
    user_dict[feature_dict[user]] = {'name':feature_dict[user], 'attribute':[]}

item_dict = {}
for item_cat in merge_pd['item_cat'].unique():
    item, cat = item_cat.split('_')
    item_dict[feature_dict[item]] = {'title':feature_dict[item], 'attribute':[feature_dict[cat]]}

pk.dump(feature_dict, open(f'{folder}/feature_dict.pkl', 'wb'))
pk.dump(user_dict, open(f'{folder}/user_dict.pkl', 'wb'))
pk.dump(item_dict, open(f'{folder}/item_dict.pkl', 'wb'))

print('done')