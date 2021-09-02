import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch.utils.data as data
import pickle
import config


# def load_all(test_num=100):
#   """ We load all the three file here to save time in each epoch. """
#   train_data = pd.read_csv(
#       config.train_rating, 
#       sep='\t', header=None, names=['user', 'item'], 
#       usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

#   user_num = train_data['user'].max() + 1
#   item_num = train_data['item'].max() + 1

#   train_data = train_data.values.tolist()

#   # load ratings as a dok matrix
#   train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
#   for x in train_data:
#       train_mat[x[0], x[1]] = 1.0

#   test_data = []
#   with open(config.test_negative, 'r') as fd:
#       line = fd.readline()
#       while line != None and line != '':
#           arr = line.split('\t')
#           u = eval(arr[0])[0]
#           test_data.append([u, eval(arr[0])[1]])
#           for i in arr[1:]:
#               test_data.append([u, int(i)])
#           line = fd.readline()

#   return train_data, test_data, user_num, item_num, train_mat

def binarize_dataset(threshold, training_users, training_items, training_ratings):
    for i in range(len(training_ratings)):
        if training_ratings[i] > threshold:
            training_ratings[i] = 1
        else:
            training_ratings[i] = 0
    training_users = [training_users[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_items = [training_items[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_ratings = [rating for rating in training_ratings if rating != 0]
    return training_users, training_items, training_ratings

def load_all(test_num=100):
    if config.dataset in ["amazon-book20", "yelp4"]:
        train_file = config.main_path + config.dataset + '/train.pkl'
        test_file = config.main_path + config.dataset + '/test.pkl'

        train = pickle.load(open(train_file, "rb"))
        train_data = []
        for user, items in train.items():
            for item in items:
                train_data.append([user, item])

        train = np.array(train_data).astype(int)
        user_num = np.max(train[:, 0])
        item_num = np.max(train[:, 1])

        test = pickle.load(open(test_file, "rb"))
        test_data = []
        for user, items in test.items():
            for item in items:
                test_data.append([user, item])

        test = np.array(test_data).astype(int)
        user_num = max(user_num, np.max(test[:, 0])) + 1
        item_num = max(item_num, np.max(test[:, 1])) + 1

        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data:
            train_mat[x[0], x[1]] = 1.0

    elif config.dataset in ["TAFA-digital-music", "TAFA-grocery"]:
        train_file = config.main_path + config.dataset + '/train.pkl'
        test_file = config.main_path + config.dataset + '/val.pkl'

        train = pickle.load(open(train_file, "rb"))
        train_users, train_items, train_ratings = train
        train_users, train_items, train_ratings = binarize_dataset(3, train_users, train_items,
                                                                       train_ratings)
        train_data = []
        for uid, iid in zip(train_users, train_items):
            train_data.append([uid, iid])

        train = np.array(train_data).astype(int)
        user_num = np.max(train[:, 0])
        item_num = np.max(train[:, 1])

        test = pickle.load(open(test_file, "rb"))
        test_users, test_items, test_ratings = test
        test_users, test_items, test_ratings = binarize_dataset(3, test_users, test_items,
                                                                 test_ratings)
        test_data = []
        for uid, iid in zip(test_users, test_items):
            test_data.append([uid, iid])

        test = np.array(test_data).astype(int)
        user_num = max(user_num, np.max(test[:, 0])) + 1
        item_num = max(item_num, np.max(test[:, 1])) + 1

        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data:
            train_mat[x[0], x[1]] = 1.0

    return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features, 
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
                    else self.features_ps
        labels = self.labels_fill if self.is_training \
                    else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item ,label
        