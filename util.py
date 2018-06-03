import seaborn as sns
import pandas as pd
import common
import os
import matplotlib.pyplot as plt

def concat_times(dir, csv_name='train_data.csv'):
    # subfolders are 0, 1, 2, ...
    subfolders = next(os.walk(dir))[1]
    frames = []
    for i, subfolder in enumerate(subfolders):
        df = pd.read_csv('{}/{}/{}'.format(dir, subfolder, csv_name)).drop(['Unnamed: 0'], axis=1)
        #df[common.S_TIMES] = i
        df[common.S_TIMES] = subfolder
        frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    return res

def concat_models(dir, csv_name='train_data.csv'):
    # subfolders are ddpg, dqn, ppo, ...
    # subsubfolders are ddpg/0, ddpg/1, ddpg/2, ...
    subfolders = next(os.walk(dir))[1]
    frames = []
    for subfolder in subfolders:
        df = concat_times('{}/{}'.format(dir, subfolder), csv_name)
        df[common.S_MODEL] = subfolder
        frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    return res

def plot(df, dir='.', name='sum.png', ylim=None, **kwargs):
    plt.figure()
    headers = list(df)
    unit = None
    condition = None
    if common.S_TIMES in headers:
        unit = common.S_TIMES
    if common.S_MODEL in headers:
        condition = common.S_MODEL
    if ylim is not None:
        sns.plt.ylim(ylim)
    sns_plot = sns.tsplot(data=df, time=common.S_EPI, value=common.S_TOTAL_R, unit=unit, condition=condition, **kwargs)
    plt.savefig('{}/{}'.format(dir, name), dpi=200)
    plt.close()

def plot_custom(df, unit=None, condition=None, dir='.', name='sum.png', ylim=None, **kwargs):
    plt.figure()
    headers = list(df)
    if ylim is not None:
        sns.plt.ylim(ylim)
    sns_plot = sns.tsplot(data=df, time=common.S_EPI, value=common.S_TOTAL_R, unit=unit, condition=condition, **kwargs)
    plt.savefig('{}/{}'.format(dir, name), dpi=200)
    plt.close()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data = target_param.data * (1.0 - tau) + param.data * tau

def hard_update(target, source):
    target.load_state_dict(source.state_dict())
