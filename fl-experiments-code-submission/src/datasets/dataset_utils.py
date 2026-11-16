import os
import ujson
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

batch_size = 10
train_size = 0.75 
least_samples = batch_size / (1-train_size) 
alpha = 0.1 

class AdultIncomeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        skipped_rows = []
        self.data = pd.read_csv(csv_file, sep=',', skiprows=1, on_bad_lines=lambda x: skipped_rows.append(x), engine='python')

        self.data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                             'native-country', 'income']
        self.transform = transform

        self.data.replace('?', pd.NA, inplace=True)

        missing_values = self.data.isnull().sum()

        self.data.dropna(inplace=True)

        self.label_encoders = {}
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country', 'income']

        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col].astype(str))

        numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.scaler = StandardScaler()
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])

        self.X = self.data.drop('income', axis=1).values
        self.y = self.data['income'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.X[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    '''
    Distribute datapoints:
    1. IID if (niid, partition, class_per_client) = (False, 'pat', num_classes).
    2. Non-IID via Pathological Splits if (True, 'pat', class_per_client).
        This extreme form of non-iid distribution results in each client receiving data from only a few classes.
    3. Non-IID via Dirichlet Distribution if (True, 'dir'), with lower alpha creating more uneven splits.
        This is a more controlled data splitting method, lower alpha -> more uneven (highly non-iid) distribution.

    Inputs:
    1. data : dataset contents and data labels
    2. num_clients (int): number of all clients in system
    3. num_classes (int): number of classes (distinct labels) within the dataset
    4. niid (bool): boolean that specifies if the data should be distributed non-iid (True) or iid (False)
    5. balance (bool): boolean that specifies if the total number of data points per client is fixed or not (False).
    6. partition (string): whether data is non-iid distributed based on Dirichlet distribution or Pathological Splits
    7. class_per_client (int): number of classes in each client's local dataset

    Outputs:
    1. X (list of dataset content): the i-th entry contains the dataset content distributed to client i
    2. y (list of dataset labels): the i-th entry contains the corresponding dataset labels distributed to client i
    3. statistic (list): The i-th entry shows the number of data points per label for client i.
    '''
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    # distribute data via Pathological Splits
    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    # distribute data via Dirichlet Distribution
    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        if num_classes == 2:
            alpha = 0.7
        else:
            alpha = 0.1

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

        # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic



def split_data(X, y):
    '''
    Distribute train dataset and test dataset based on assigned data contents and data labels to each client

    Inputs:
    1. X (list of dataset content): the i-th entry contains the dataset content distributed to client i
    2. y (list of dataset labels): the i-th entry contains the corresponding dataset labels distributed to client i

    Outputs:
    train_data (list of train data points): the i-th entry contains the assigned training data points to client i
    test_data (list of test data points): the i-th entry contains the assigned test data points to client i
    '''
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, test_data




def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, balance=True, partition=None):
    '''check if dataset already exists and is distributed across clients'''
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")



def read_data(dataset_name, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('./data/', dataset_name, 'train/').replace("\\", "/")

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('./data/', dataset_name, 'test/').replace("\\", "/")

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data