import argparse

control_args = ['gpu', 'path', 'env', 'repeat', 'n_test', 'manual', 'save_interval', 'load']
model_args = ['mem_size', 'lr_critic', 'lr_actor', 'epsilon', 'max_epi', 'epsilon_decay',
        'gamma', 'target_update_frequency', 'batch_size', 'random_process', 'max_step']

def get_args():
    parser = argparse.ArgumentParser(description='rl')
    # control args
    parser.add_argument('--gpu', type=str, choices=['0', '1', '2', '3'], required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--manual', type=bool, default=False)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--load', type=str)
    
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
    parser.add_argument('--random_process', type=bool)
    parser.add_argument('--max_step', type=int)

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
