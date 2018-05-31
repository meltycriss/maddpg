import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

control_args = ['gpu', 'path', 'env', 'repeat', 'n_test', 'manual', 'save_interval', 'load', 'env_normalized', 'train']
model_args = ['mem_size', 'lr_critic', 'lr_actor', 'epsilon', 'max_epi', 'epsilon_decay',
        'gamma', 'target_update_frequency', 'batch_size', 'random_process_mode', 'max_step', 
        'actor_update_mode', 'popart', 'actor', 'critic', 'epsilon_start', 'epsilon_end', 
        'epsilon_rate', 'partition_num']

def get_args():
    parser = argparse.ArgumentParser(description='rl')
    # control args
    parser.add_argument('--gpu', type=str, choices=['0', '1', '2', '3'], required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--manual', type=str2bool, default=False)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--load', type=str)
    parser.add_argument('--env_normalized', type=str2bool, default=True)
    parser.add_argument('--train', type=str2bool, default=True)
    
    ##############################################
    # remember to change global control_args
    ##############################################

    # model args
    parser.add_argument('--mem_size', type=int)
    parser.add_argument('--lr_critic', type=float)
    parser.add_argument('--lr_actor', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--max_epi', type=int)
    parser.add_argument('--epsilon_decay', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--target_update_frequency', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--random_process_mode', type=str)
    parser.add_argument('--max_step', type=int)
    parser.add_argument('--actor_update_mode', type=str, choices=['default', 'dynamic', 'obo', 'obo_target'])
    parser.add_argument('--popart', type=str2bool)
    parser.add_argument('--actor', type=str)
    parser.add_argument('--critic', type=str)
    parser.add_argument('--epsilon_start', type=float)
    parser.add_argument('--epsilon_end', type=float)
    parser.add_argument('--epsilon_rate', type=float)
    parser.add_argument('--partition_num', type=int)

    ##############################################
    # remember to change global model_args
    ##############################################

    args = parser.parse_args()
    return args

def get_model_args():
    args = get_args()
    res = vars(args)
    for key, value in res.items():
        if key in control_args:
            res.pop(key)
            continue
        if value is None:
            res.pop(key)
            continue
    return res

def get_control_args():
    args = get_args()
    res = vars(args)
    for key, value in res.items():
        if key in model_args:
            res.pop(key)
            continue
        if value is None:
            res.pop(key)
            continue
    return res
