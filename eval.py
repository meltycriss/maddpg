from tqdm import trange
import os
import util
import common
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-l', '--level', type=str, choices=['model', 'repeat'], required=True)
    parser.add_argument('-i', '--interval', type=int, nargs='+', default=[1, 10, 100])
    args = parser.parse_args()
    res = vars(args)
    return res

def model_level(root, intervals):
    # model level
    # train data
    df = util.concat_models(root, csv_name='train_data.csv')
    for interval in intervals:
        util.plot(df[df[common.S_EPI] % interval == 0], dir=root, name='train_data_gap_{}.png'.format(interval))
    # test data
    df = util.concat_models(root, csv_name='test_data.csv')
    util.plot(df[df[common.S_EPI] >= 0], dir=root, name='test_data.png')
    
    # repeat level
    subfolders = next(os.walk(root))[1]
    for subfolder in subfolders:
        path = os.path.join(root, subfolder)
        repeat_level(path, intervals)

def repeat_level(root, intervals):
    # train data
    df = util.concat_times(root, csv_name='train_data.csv')
    df[common.S_MODEL] = os.path.basename(root)
    # test data
    for interval in intervals:
        util.plot(df[df[common.S_EPI] % interval == 0], dir=root, name='train_data_gap_{}.png'.format(interval))
    df = util.concat_times(root, csv_name='test_data.csv')
    df[common.S_MODEL] = os.path.basename(root)
    util.plot(df[df[common.S_EPI] >= 0], dir=root, name='test_data.png')

args = get_args()
if args['level']=='model':
    model_level(args['path'], args['interval'])
elif args['level']=='repeat':
    repeat_level(args['path'], args['interval'])




