import gym
import gym_foa
from maddpg import MADDPG
from tqdm import trange
import os
import util
import common
import logging
import arguments

# suppress INFO level logging 'Starting new video recorder writing to ...'
logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.WARNING)
# suppress INFO level logging 'Creating monitor directory ...'
logging.getLogger('gym.wrappers.monitoring').setLevel(logging.WARNING)


control_args = arguments.get_control_args()

ENV_NAME = control_args['env']
env = gym.make(ENV_NAME)

os.environ["CUDA_VISIBLE_DEVICES"] = control_args['gpu']

root = control_args['path']
if not os.path.exists(root):
    os.mkdir(root)

args = None
if control_args['manual']:
    args = [
            {'max_epi':20000, 'mem_size':10000},
            {'max_epi':20000, 'mem_size':100000},
            {'max_epi':20000, 'mem_size':1000000},
            ]
else:
    model_args = arguments.get_model_args()
    args = [
            model_args,
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
    for n in trange(control_args['repeat'], desc='repeat', leave=True):
        dir = '{}/{}'.format(model_dir, n)
        maddpg=MADDPG(env, **arg)
        if control_args.has_key('load'):
            model_path = control_args['load']
            maddpg.load_actor(os.path.join(model_path, 'actor.pt'))
            maddpg.load_critic(os.path.join(model_path, 'critic.pt'))
        maddpg.train(dir, control_args['save_interval'])
        maddpg.save(dir)
        maddpg.test(dir, n=control_args['n_test'])





