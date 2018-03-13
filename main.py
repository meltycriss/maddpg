import gym
import gym_foa
from ddpg import DDPG
from tqdm import trange
import os
import shutil
import util
import common
import sys
import logging

# suppress INFO level logging 'Starting new video recorder writing to ...'
logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.WARNING)
# suppress INFO level logging 'Creating monitor directory ...'
logging.getLogger('gym.wrappers.monitoring').setLevel(logging.WARNING)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

ENV_NAME = 'foa-v0'
env = gym.make(ENV_NAME)

root = ENV_NAME
if len(sys.argv)>1:
    root = sys.argv[1]

if os.path.exists(root):
    shutil.rmtree(root)
os.mkdir(root)

args = [
        {},
        #{'max_epi':3},
        #{'random_process':False}
        ]
# model_names is a list like [max_epi_10, max_epi_20]
model_names = [
        '_'.join(
        [
        '_'.join(
        [key, str(value)])
        for key, value in arg.items()])
        for arg in args]
# handle standard arg, i.e., {}
model_names = ['standard' if name=='' else name  for name in model_names]
    

# model loop
for i in trange(len(args), desc='model', leave=True):
    model_dir = '{}/{}'.format(root, model_names[i])
    os.mkdir(model_dir)
    arg = args[i]
    # repeat loop
    for n in trange(2, desc='repeat', leave=True):
        dir = '{}/{}'.format(model_dir, n)
        ddpg=DDPG(env, **arg)
        ddpg.train(dir)
        ddpg.save(dir)
        ddpg.test(dir, n=100)

# visualization
# train data
df = util.concat_models(root, csv_name='train_data.csv')
util.plot(df[df[common.S_EPI] > 0], dir=root, name='train_data.png')

# test data
df = util.concat_models(root, csv_name='test_data.csv')
util.plot(df[df[common.S_EPI] > 0], dir=root, name='test_data.png')




