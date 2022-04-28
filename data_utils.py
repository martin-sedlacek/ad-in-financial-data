import numpy as np
import torch
import pandas as pd
import torch.utils.data as data
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from os import path, listdir


'''***************************************************************************************
*   Financial data sources
***************************************************************************************
*    Title: Daily News for Stock Market Prediction
*    Availability: https://www.kaggle.com/datasets/aaron7sun/stocknews?ref=hackernoon.com
*    Author: AARON7SUN (Kaggle alias)
*    Date: 2020
*    Associated data: 
*       - data/financial_data/RedditNews.csv
*       - data/financial_data/Combined_DJIA.csv
***************************************************************************************
*    Title: Huge Stock Market Dataset
*    Availability: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?ref=hackernoon.com
*    Author: BORIS MARJANOVIC (Kaggle alias)
*    Date: 2018
*    Associated data: 
*       - data/financial_data/ETFs/
*       - data/financial_data/Stocks/
***************************************************************************************
*    Title: S&P 500 stock data
*    Availability: https://www.kaggle.com/datasets/camnugent/sandp500
*    Author: CAM NUGENT (Kaggle alias)
*    Date: 2018
*    Associated data: 
*       - data/financial_data/spx.csv
***************************************************************************************'''


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
    return pd.read_csv("data/financial_data/normalised_result.csv")


def load_sentiment_data():
    df = pd.read_csv("data/financial_data/Combined_News_DJIA.csv", usecols=['Date', 'Label'])

    # Set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Rename column
    df.columns = ['Sentiment']
    return df


def load_spx_data(raw=False):
    df = pd.read_csv("data/financial_data/spx.csv")

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
        c, stocks = load_txts('data/financial_data/Stocks/', c, raw=raw)
        c, etfs = load_txts('data/financial_data/ETFs/', c, raw=raw)
    else:
        _, stocks = load_txts('data/financial_data/Stocks/', raw=raw)
        _, etfs = load_txts('data/financial_data/ETFs/', raw=raw)

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


def load_stock_as_crossvalidated_timeseries(file_path, seq_length, seq_stride, gen_seq_len, bs, num_features=7, normalise=True, load_as_dl=True):
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

    samples = samples.drop(columns=['Ticker'])
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SPX_Close']

    for c in columns:
        samples[c] = samples[c].pct_change()

    samples = drop_non_numeric(samples)

    if normalise:
        for c in columns:
            samples[c] = (samples[c] - samples[c].mean()) / samples[c].std()

    samples = samples.values

    num_samples_t = (samples.shape[0] - seq_length - seq_length) // seq_stride
    prev_steps = np.empty([num_samples_t, seq_length, num_features])
    next_steps = np.empty([num_samples_t, gen_seq_len, num_features])
    for j in range(num_samples_t):
        for i in range(num_features):
            prev_steps[j, :, i] = samples[(j * seq_stride):(j * seq_stride + seq_length), i]
            next_steps[j, :, i] = samples[(j * seq_stride + seq_length):(j * seq_stride + seq_length + gen_seq_len), i]

    prev_steps = torch.tensor(prev_steps.astype(np.float32))
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
        x_train = prev_steps[train_range[0]:train_range[1]].clone()
        p_train = next_steps[train_range[0]:train_range[1]].clone()
        x_test = prev_steps[test_range[0]:test_range[1]].clone()
        p_test = next_steps[test_range[0]:test_range[1]].clone()
        if load_as_dl:
            train_data = data.TensorDataset(x_train, p_train)
            test_data = data.TensorDataset(x_test, p_test)
            training_iter = data.DataLoader(train_data, bs, shuffle=False, num_workers=2)
            testing_iter = data.DataLoader(test_data, bs, shuffle=False, num_workers=2)
            tscv_dl_list.append((training_iter, testing_iter))
        else:
            tscv_dl_list.append((x_train.numpy(), p_train.numpy(), x_test.numpy(), p_test.numpy()))
    return tscv_dl_list


'''***************************************************************************************
*   KDD99 dataset sources 
***************************************************************************************
*    Paper title: MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks
*    Availability: https://github.com/LiDan456/MAD-GANs; https://arxiv.org/pdf/1901.04997.pdf
*    Author: Li et al.
*    Date: Jan 17, 2019
*    Associated data: 
*       - data/kdd99/kdd99_test.npy
*       - data/kdd99/kdd99_train.npy
***************************************************************************************'''


# ***************************************************************************************
# The load_kdd99 method contains parts inspired by the original code for data load associated with Li et al. (2019)
# Availability: https://github.com/LiDan456/MAD-GANs; https://arxiv.org/pdf/1901.04997.pdf
# ***************************************************************************************
def load_kdd99(file_path, seq_length, seq_step, num_signals, gen_seq_len): #, bs, deepant=False
    dataset = np.load(file_path)

    m, n = dataset.shape
    for i in range(n - 1):
        B = max(dataset[:, i])
        if B != 0:
            dataset[:, i] /= max(dataset[:, i])
            dataset[:, i] = 2 * dataset[:, i] - 1
        else:
            dataset[:, i] = dataset[:, i]

    samples = dataset[:, 0:n - 1]
    labels = dataset[:, n - 1]

    # apply PCA dimension reduction for multi-variate data
    X_a = samples
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))
    samples = T_a

    # Generate datasets
    num_samples_t = (samples.shape[0] - seq_length - gen_seq_len) // seq_step
    prev_steps = np.empty([num_samples_t, seq_length, num_signals])
    prev_steps_labels = np.empty([num_samples_t, seq_length, 1])
    next_steps = np.empty([num_samples_t, gen_seq_len, num_signals])
    next_steps_labels = np.empty([num_samples_t, gen_seq_len, 1])

    for j in range(num_samples_t):
        prev_steps_labels[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        next_steps_labels[j, :, :] = np.reshape(labels[(j * seq_step + seq_length):(j * seq_step + seq_length + gen_seq_len)], [-1, 1])
        for i in range(num_signals):
            prev_steps[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
            next_steps[j, :, i] = samples[(j * seq_step + seq_length):(j * seq_step + seq_length + gen_seq_len), i]

    '''if deepant:
        ll = [prev_steps, prev_steps_labels, next_steps, next_steps_labels]
        ll = [torch.tensor(x.astype(np.float32)) for x in ll]
        ds = data.TensorDataset(ll[0], ll[1], ll[2], ll[3])
    else:
        ll = [prev_steps, prev_steps_labels]
        ll = [torch.tensor(x.astype(np.float32)) for x in ll]
        ds = data.TensorDataset(ll[0], ll[1])

    return data.DataLoader(ds, shuffle=False, batch_size=bs)'''
    return [prev_steps, prev_steps_labels, next_steps, next_steps_labels]


def make_kdd99_dl(ll, bs, deepant=False):
    ll = [torch.tensor(x.astype(np.float32)) for x in ll]

    if deepant:
        #ll = [prev_steps, prev_steps_labels, next_steps, next_steps_labels]
        #ll = [torch.tensor(x.astype(np.float32)) for x in ll]
        ds = data.TensorDataset(ll[0], ll[1], ll[2], ll[3])
    else:
        #ll = [prev_steps, prev_steps_labels]
        #ll = [torch.tensor(x.astype(np.float32)) for x in ll]
        ds = data.TensorDataset(ll[0], ll[1])

    return data.DataLoader(ds, shuffle=False, batch_size=bs)


def kdd99(seq_length, seq_stride, num_generated_features, gen_seq_len, batch_size, deepant=False):
    train_ll = load_kdd99('data/kdd99/kdd99_train.npy', seq_length, seq_stride, num_generated_features, gen_seq_len)
    test_ll = load_kdd99('data/kdd99/kdd99_test.npy', seq_length, seq_stride, num_generated_features, gen_seq_len)
    train_dl = make_kdd99_dl(train_ll, batch_size, deepant)
    test_dl = make_kdd99_dl(test_ll, batch_size, deepant)
    return train_dl, test_dl