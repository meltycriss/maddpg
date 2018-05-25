from tqdm import trange
import os
import util
import common
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(x):
    res = float(x)
    return res

def get_args():
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-l', '--level', type=str, choices=['model', 'repeat'], required=True)
    parser.add_argument('-i', '--interval', type=int, nargs='+', default=[1, 10, 100])
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--test', type=str2bool, default=True)
    parser.add_argument('--ylim', type=str2float, nargs='+', default=None)
    args = parser.parse_args()
    res = vars(args)
    return res

def model_level(root, intervals, train, test, ylim):
    # model level
    # train data
    if train:
        df = util.concat_models(root, csv_name='train_data.csv')
        for interval in intervals:
            util.plot(df[df[common.S_EPI] % interval == 0], dir=root, name='train_data_gap_{}.png'.format(interval), ylim=ylim)
    if test:
        # test data
        df = util.concat_models(root, csv_name='test_data.csv')
        util.plot(df[df[common.S_EPI] >= 0], dir=root, name='test_data.png', ylim=ylim)
    
    # repeat level
    subfolders = next(os.walk(root))[1]
    for subfolder in subfolders:
        path = os.path.join(root, subfolder)
        repeat_level(path, intervals, train, test, ylim)

def repeat_level(root, intervals, train, test, ylim):
    # train data
    if train:
        df = util.concat_times(root, csv_name='train_data.csv')
        df[common.S_MODEL] = os.path.basename(root)
        for interval in intervals:
            util.plot(df[df[common.S_EPI] % interval == 0], dir=root, name='train_data_gap_{}.png'.format(interval), ylim=ylim)
            util.plot_custom(df[df[common.S_EPI] % interval == 0], condition=common.S_TIMES, unit=common.S_MODEL, dir=root, name='train_data_sep_gap_{}.png'.format(interval), ylim=ylim)
    if test:
        # test data
        df = util.concat_times(root, csv_name='test_data.csv')
        df[common.S_MODEL] = os.path.basename(root)
        util.plot(df[df[common.S_EPI] >= 0], dir=root, name='test_data.png', ylim=ylim)
        util.plot_custom(df[df[common.S_EPI] >= 0], condition=common.S_TIMES, unit=common.S_MODEL, dir=root, name='test_data_sep.png', ylim=ylim)

args = get_args()
if args['level']=='model':
    model_level(args['path'], args['interval'], args['train'], args['test'], args['ylim'])
elif args['level']=='repeat':
    repeat_level(args['path'], args['interval'], args['train'], args['test'], args['ylim'])




