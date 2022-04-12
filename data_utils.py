import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

def kdd99(seq_length, seq_step, num_signals):
    train = np.load('data/kdd99_train.npy')
    print('load kdd99_train from .npy')
    m, n = train.shape  # m=562387, n=35
    # normalization
    for i in range(n - 1):
        # print('i=', i)
        A = max(train[:, i])
        # print('A=', A)
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[:, 0:n - 1]
    labels = train[:, n - 1]  # the last colummn is label
    #############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    #from sklearn.decomposition import PCA
    X_n = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=6 -- #
    n_components = num_signals
    pca = PCA(n_components, svd_solver='full')
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_
    # projected values on the principal component
    T_n = np.matmul(X_n, pc.transpose(1, 0))
    samples = T_n
    # # only for one-dimensional
    # samples = T_n.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    num_samples = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])
    next_steps = np.empty([num_samples, num_signals])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
            next_steps[j, i] = samples[(j * seq_step + seq_length)+1, i]

    samples = aa
    labels = bb

    return samples, labels, next_steps


def kdd99_test(seq_length, seq_step, num_signals):
    test = np.load('data/kdd99_test.npy')
    print('load kdd99_test from .npy')

    m, n = test.shape  # m1=494021, n1=35

    for i in range(n - 1):
        B = max(test[:, i])
        if B != 0:
            test[:, i] /= max(test[:, i])
            # scale from -1 to 1
            test[:, i] = 2 * test[:, i] - 1
        else:
            test[:, i] = test[:, i]

    samples = test[:, 0:n - 1]
    labels = test[:, n - 1]
    idx = np.asarray(list(range(0, m)))  # record the idx of each point
    #############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    #from sklearn.decomposition import PCA
    X_a = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=6 -- #
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))
    samples = T_a
    # # only for one-dimensional
    # samples = T_a.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    #bb = np.empty([num_samples_t, seq_length, 1])
    bb = np.empty([num_samples_t, 1])
    #bbb = np.empty([num_samples_t, seq_length, 1])
    next_steps = np.empty([num_samples_t, num_signals])
    for j in range(num_samples_t):
        #bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        #bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bb[j] = labels[(j * seq_step + seq_length)+1]
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
            next_steps[j, i] = samples[(j * seq_step + seq_length)+1, i]

    samples = aa
    labels = bb
    #index = bbb

    return samples, labels, next_steps


def load_kdd99(seq_length, seq_stride, num_generated_features, batch_size):
    train_samples, train_labels, train_preds = kdd99(seq_length, seq_stride, num_generated_features)
    test_samples, test_labels, test_preds = kdd99_test(seq_length, seq_stride, num_generated_features)

    train_x = torch.Tensor(train_samples)
    train_y = torch.Tensor(train_labels)
    train_preds = torch.Tensor(train_preds)
    train_data = torch.utils.data.TensorDataset(train_x, train_y, train_preds)
    train_dl = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=batch_size)

    test_x = torch.Tensor(test_samples)
    test_y = torch.Tensor(test_labels)
    test_preds = torch.Tensor(test_preds)
    test_data = torch.utils.data.TensorDataset(test_x, test_y, test_preds)
    test_dl = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)
    return train_dl, test_dl


def load_financial(batch_size):
    normalised_data = pd.read_csv("data/normalised_result.csv")

    # Date = How many days since 2008-09-02?
    # Date encoding this is a fairly slow transformation and you need let it run for a while
    normalised_data['Date'] = normalised_data['Date'].apply(lambda x: int((datetime.strptime(x,
                                                                                             '%Y-%m-%d') - datetime.strptime(
        '2008-09-02', '%Y-%m-%d')).total_seconds() / 60 / 60 / 24))
    # Convert to tensor
    np_array = normalised_data.to_numpy()
    tensor_data = torch.tensor(normalised_data.values)

    # Generate forward chaining cross-validation ranges
    tscv = TimeSeriesSplit()
    split_indices = tscv.split(normalised_data)
    train_indice_ranges = []
    test_indice_ranges = []
    for train_index, test_index in split_indices:
        train_indice_ranges.append((0, train_index[len(train_index) - 1]))
        test_indice_ranges.append((test_index[0], test_index[len(test_index) - 1]))

    # Initialize data loaders
    data_loaders = []
    for fold in range(len(train_indice_ranges)):
        train_range = train_indice_ranges[fold]
        test_range = test_indice_ranges[fold]
        training_iter = torch.utils.data.DataLoader(tensor_data[test_range[0]:test_range[1]], batch_size,
                                                    shuffle=False, num_workers=2)
        testing_iter = torch.utils.data.DataLoader(tensor_data[train_range[0]:train_range[1]], batch_size,
                                                   shuffle=False, num_workers=2)
        data_loaders.append((training_iter, testing_iter))
    return data_loaders
