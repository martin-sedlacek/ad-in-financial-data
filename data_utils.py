import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from os import path, listdir


def kdd99(seq_length, seq_step, num_signals, gen_seq_len):
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

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    # -- the best PC dimension is chosen pc=6 -- #
    X_n = samples
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
    num_samples = (samples.shape[0] - seq_length - gen_seq_len) // seq_step
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])
    next_steps = np.empty([num_samples, gen_seq_len, num_signals])
    next_steps_labels = np.empty([num_samples, gen_seq_len, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        next_steps_labels[j, :, :] = np.reshape(labels[(j * seq_step + seq_length):(j * seq_step + seq_length + gen_seq_len)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
            next_steps[j, :, i] = samples[(j * seq_step + seq_length):(j * seq_step + seq_length + gen_seq_len), i]

    return aa, bb, next_steps, next_steps_labels


def kdd99_test(seq_length, seq_step, num_signals, gen_seq_len):
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

    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    # -- the best PC dimension is chosen pc=6 -- #
    X_a = samples
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))
    samples = T_a
    # # only for one-dimensional
    # samples = T_a.reshape([samples.shape[0], ])
    num_samples_t = (samples.shape[0] - seq_length - gen_seq_len) // seq_step

    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    next_steps = np.empty([num_samples_t, gen_seq_len, num_signals])
    next_steps_labels = np.empty([num_samples_t, gen_seq_len, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        next_steps_labels[j, :, :] = np.reshape(labels[(j * seq_step + seq_length):(j * seq_step + seq_length + gen_seq_len)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
            next_steps[j, :, i] = samples[(j * seq_step + seq_length):(j * seq_step + seq_length + gen_seq_len), i]

    return aa, bb, next_steps, next_steps_labels


def load_kdd99(seq_length, seq_stride, num_generated_features, gen_seq_len, batch_size):
    train_samples, train_labels, train_preds, train_preds_labels = kdd99(seq_length, seq_stride, num_generated_features, gen_seq_len)
    test_samples, test_labels, test_preds, test_preds_labels = kdd99_test(seq_length, seq_stride, num_generated_features, gen_seq_len)

    def make_dl(x, y, p, pb, bs):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        p = torch.Tensor(p)
        pb = torch.Tensor(pb)
        data = torch.utils.data.TensorDataset(x, y, p, pb)
        return torch.utils.data.DataLoader(data, shuffle=False, batch_size=bs)

    train_dl = make_dl(train_samples, train_labels, train_preds, train_preds_labels, batch_size)
    test_dl = make_dl(test_samples, test_labels, test_preds, test_preds_labels, batch_size)
    return train_dl, test_dl


def load_sentiment_data():
    data = pd.read_csv("./data/Combined_News_DJIA.csv", usecols=['Date', 'Label'])

    # Set Date as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Rename column
    data.columns = ['Sentiment']
    return data


def load_spx_data(raw=False):
    data = pd.read_csv("./data/spx.csv")

    # Set Date as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Rename column
    data.columns = ['SPX_Close']

    if not raw:
        # Relative change features
        data["SPX_Close"] = data["SPX_Close"].pct_change()

    return data


def load_stock(file_path, ticker_code=-1):
    df = pd.read_csv(file_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if ticker_code < 0:
        df['Ticker'] = path.basename(file_path).replace('.txt', '')
    else:
        df['Ticker'] = ticker_code
    return df

def load_txts(dir_path, ctr=-1, raw=False):
    # Load the data for multiple .txt files
    data = []
    csv_paths = [dir_path + x for x in listdir(dir_path) if x.endswith('.txt') and path.getsize(dir_path + x) > 0]
    for file_path in csv_paths:
        df = load_stock(file_path, ctr)
        if not ctr < 0:
            ctr += 1
        data.append(df)
    data = pd.concat(data, ignore_index=True)
    data.reset_index(inplace=True, drop=True)

    # Set Date as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    if not raw:
        # Relative change features
        data["Open"] = data["Open"].pct_change()
        data["Close"] = data["Close"].pct_change()
        data["High"] = data["High"].pct_change()
        data["Low"] = data["Low"].pct_change()
        data["Volume"] = data["Volume"].pct_change()

    return ctr, data


def load_financial(raw=False, encode_ticker=True):
    # Load initial data
    sentiment = load_sentiment_data()
    spx = load_spx_data(raw=raw)

    if encode_ticker:
        c = 0
        c, stocks = load_txts('./data/Stocks/', c, raw=raw)
        c, etfs = load_txts('./data/ETFs/', c, raw=raw)
    else:
        _, stocks = load_txts('./data/Stocks/', raw=raw)
        _, etfs = load_txts('./data/ETFs/', raw=raw)

    # Filter sentiment and spx dataframes by time
    start, end = '2008-09-01', '2016-07-01'
    sentiment = filter_by_time(sentiment, start, end)
    spx = filter_by_time(spx, start, end)

    # Join the datasets into a single dataframe "result"
    stock_etf_df = pd.concat([etfs, stocks])
    stock_etf_df = filter_by_time(stock_etf_df, start, end)
    sentiment_labeled_df = stock_etf_df.join(sentiment, how="outer")
    result = sentiment_labeled_df.join(spx, how="outer")

    # drop NA and inf values
    result = drop_non_numeric(result)

    # apply Z-normalisation if raw flag is not set
    if not raw:
        result = z_normalisation(result)

    return result


def filter_by_time(df, start, end):
    return df.sort_index().loc[start:end]


def drop_non_numeric(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def z_normalisation(df):
    columns = ['Open', 'Close', 'High', 'Low', 'Close', 'Volume', 'SPX_Close']
    for c in columns:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df


def encode_date(df):
    # Date = How many days since 2008-09-02?
    # Date encoding this is a fairly slow transformation and you need let it run for a while
    df['Date'] = df['Date'].apply(lambda x: int((datetime.strptime(x,'%Y-%m-%d') - datetime.strptime(
        '2008-09-02', '%Y-%m-%d')).total_seconds() / 60 / 60 / 24))
    return df


def load_financial_data():
    return pd.read_csv("data/normalised_result.csv")


def forward_chaining_crossvalidation(df, batch_size):
    tensor_data = torch.tensor(df.values)

    # Generate forward chaining cross-validation ranges
    tscv = TimeSeriesSplit()
    split_indices = tscv.split(df)
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