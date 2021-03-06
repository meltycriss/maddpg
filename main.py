import gym
import gym_foa
from maddpg import MADDPG
from tqdm import trange
import os
import util
import common
import logging
import arguments
from normalized_env import NormalizedEnv

# suppress INFO level logging 'Starting new video recorder writing to ...'
logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.WARNING)
# suppress INFO level logging 'Creating monitor directory ...'
logging.getLogger('gym.wrappers.monitoring').setLevel(logging.WARNING)


control_args = arguments.get_control_args()

ENV_NAME = control_args['env']
env = gym.make(ENV_NAME)
if control_args['env_normalized']:
    env = NormalizedEnv(env)

os.environ["CUDA_VISIBLE_DEVICES"] = control_args['gpu']

root = control_args['path']
if not os.path.exists(root):
    os.mkdir(root)

args = None
if control_args['manual']:
    args = [
            {'actor_update_mode':'default',},
            {'actor_update_mode':'obo',},
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
# handle invalid dir char
for i in range(len(model_names)):
    model_names[i] = model_names[i].replace('[', '').replace(']', '').replace(' ', '').replace(',', '_')
# handle standard arg, i.e., {}
model_names = ['standard' if name=='' else name  for name in model_names]

# model loop
for i in trange(len(args), desc='model', leave=True):
    model_dir = '{}/{}'.format(root, model_names[i])
    os.mkdir(model_dir)
    # log cmd
    with open('{}/cmd_config.txt'.format(model_dir), 'w') as f:
        for k, v in control_args.items():
            f.write(str(k)+': '+str(v)+'\n')
    arg = args[i]
    # repeat loop
    for n in trange(control_args['repeat'], desc='repeat', leave=True):
        dir = '{}/{}'.format(model_dir, n)
        os.mkdir(dir)
        maddpg=MADDPG(env, **arg)
        if control_args.has_key('load'):
            model_path = control_args['load']
            maddpg.load_actor(model_path)
            maddpg.load_critic(model_path)
        if control_args['train']:
            maddpg.train(dir, control_args['save_interval'])
            maddpg.save(dir)
        maddpg.test(dir, n=control_args['n_test'])





