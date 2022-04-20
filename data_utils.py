import numpy as np
import torch
import pandas as pd
import torch.utils.data as data
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from os import path, listdir

'''
KDD99
'''


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
        ds = data.TensorDataset(x, y, p, pb)
        return data.DataLoader(ds, shuffle=False, batch_size=bs)

    train_dl = make_dl(train_samples, train_labels, train_preds, train_preds_labels, batch_size)
    test_dl = make_dl(test_samples, test_labels, test_preds, test_preds_labels, batch_size)
    return train_dl, test_dl


'''
Financial data
'''


def filter_by_time(df, start, end):
    return df.sort_index().loc[start:end]


def drop_non_numeric(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def encode_date(df):
    # Date = How many days since 2008-09-02?
    # Date encoding this is a fairly slow transformation and you need let it run for a while
    df['Date'] = df['Date'].apply(lambda x: int((datetime.strptime(x,'%Y-%m-%d') - datetime.strptime(
        '2008-09-02', '%Y-%m-%d')).total_seconds() / 60 / 60 / 24))
    return df


def load_financial_data():
    return pd.read_csv("data/normalised_result.csv")


def load_sentiment_data():
    df = pd.read_csv("./data/Combined_News_DJIA.csv", usecols=['Date', 'Label'])

    # Set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Rename column
    df.columns = ['Sentiment']
    return df


def load_spx_data(raw=False):
    df = pd.read_csv("./data/spx.csv")

    # Set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Rename column
    df.columns = ['SPX_Close']

    if not raw:
        # Relative change features
        df["SPX_Close"] = df["SPX_Close"].pct_change()

    return df

def load_txts(dir_path, ctr=-1, raw=False):
    # Load the data for multiple .txt files
    ll = []
    csv_paths = [dir_path + x for x in listdir(dir_path) if x.endswith('.txt') and path.getsize(dir_path + x) > 0]
    for file_path in csv_paths:
        tmp_df = load_stock(file_path, ctr)
        if not ctr < 0:
            ctr += 1
        ll.append(tmp_df)
    df = pd.concat(ll, ignore_index=True)
    df.reset_index(inplace=True, drop=True)

    # Set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    if not raw:
        # Relative change features
        df["Open"] = df["Open"].pct_change()
        df["Close"] = df["Close"].pct_change()
        df["High"] = df["High"].pct_change()
        df["Low"] = df["Low"].pct_change()
        df["Volume"] = df["Volume"].pct_change()

    return ctr, df


def load_stock(file_path, ticker_code=-1):
    df = pd.read_csv(file_path, usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if ticker_code < 0:
        df['Ticker'] = path.basename(file_path).replace('.txt', '')
    else:
        df['Ticker'] = ticker_code
    return df


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
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SPX_Close']
    if not raw:
        for c in columns:
            result[c] = (result[c] - result[c].mean()) / result[c].std()

    return result


def load_stock_as_crossvalidated_timeseries(file_path, seq_length, seq_stride, gen_seq_len, bs, num_features=7, normalise=True):
    stock = load_stock(file_path)
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock.set_index('Date', inplace=True)

    sentiment = load_sentiment_data()
    spx = load_spx_data(raw=True)

    start, end = '2008-09-01', '2016-07-01'
    sentiment = filter_by_time(sentiment, start, end)
    spx = filter_by_time(spx, start, end)

    sentiment_labeled_df = stock.join(sentiment, how="outer")
    samples = sentiment_labeled_df.join(spx, how="outer")

    samples = samples.drop(columns=['Ticker'])  # , 'Sentiment'
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SPX_Close']

    for c in columns:
        samples[c] = samples[c].pct_change()

    samples = drop_non_numeric(samples)

    if normalise:
        for c in columns:
            samples[c] = (samples[c] - samples[c].mean()) / samples[c].std()

    samples = samples.values

    num_samples_t = (samples.shape[0] - seq_length - seq_length) // seq_stride
    aa = np.empty([num_samples_t, seq_length, num_features])
    next_steps = np.empty([num_samples_t, gen_seq_len, num_features])
    for j in range(num_samples_t):
        for i in range(num_features):
            aa[j, :, i] = samples[(j * seq_stride):(j * seq_stride + seq_length), i]
            next_steps[j, :, i] = samples[(j * seq_stride + seq_length):(j * seq_stride + seq_length + gen_seq_len), i]

    prev_steps = torch.tensor(aa.astype(np.float32))
    next_steps = torch.tensor(next_steps.astype(np.float32))

    # Generate forward chaining cross-validation ranges
    tscv = TimeSeriesSplit()
    split_indices = tscv.split(prev_steps)
    train_indice_ranges = []
    test_indice_ranges = []
    for train_index, test_index in split_indices:
        train_indice_ranges.append((0, train_index[len(train_index) - 1]))
        test_indice_ranges.append((test_index[0], test_index[len(test_index) - 1]))

    # Initialize data loaders
    tscv_dl_list = []
    for fold in range(len(train_indice_ranges)):
        train_range = train_indice_ranges[fold]
        test_range = test_indice_ranges[fold]
        x_train = torch.tensor(prev_steps[train_range[0]:train_range[1]])
        p_train = torch.tensor(next_steps[train_range[0]:train_range[1]])
        x_test = torch.tensor(prev_steps[test_range[0]:test_range[1]])
        p_test = torch.tensor(next_steps[test_range[0]:test_range[1]])
        train_data = data.TensorDataset(x_train, p_train)
        test_data = data.TensorDataset(x_test, p_test)
        training_iter = data.DataLoader(train_data, bs, shuffle=False, num_workers=2)
        testing_iter = data.DataLoader(test_data, bs, shuffle=False, num_workers=2)
        tscv_dl_list.append((training_iter, testing_iter))
