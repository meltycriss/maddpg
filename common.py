from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

Transition_agent = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'agent'))

S_EPI = 'epi'
S_TOTAL_R = 'total_r'
S_TIMES = 'times'
S_MODEL = 'model'
ATTR = [S_EPI, S_TOTAL_R]
