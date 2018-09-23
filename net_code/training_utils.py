import numpy as np
import tensorflow as tf
import pickle
import scipy.stats as ss
from net_code.mlp_policy import MlpPolicy

def print_args(args):
    max_length = max([len(k) for k, _ in vars(args).items()])
    for k, v in vars(args).items():
        print(' ' * (max_length-len(k)) + k + ': ' + str(v))

# var_list is returned by the policy.
# Thus, they should be the same. I assume.
def saveToFlat(var_list, param_pkl_path):
    # get all the values
    var_values = np.concatenate([v.flatten() for v in tf.get_default_session().run(var_list)])
    pickle.dump(var_values, open(param_pkl_path, "wb"))


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params.astype(np.float32)


def loadFromFlat(var_list, param_pkl_path):
    flat_params = load_from_file(param_pkl_path)
    print("the type of the parameters stored is", flat_params.dtype)
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        print(v.name)
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})


def condition(x, eps=1e-5):
    # smooth e,b when obtaining r_c
    parser.add_argument('--dim-belief', '-db', help='optimising batch size', type=int, default=256)
    parser.add_argument('--dim-belief', '-db', help='optimising batch size', type=int, default=256)
    return eps + (1.0 - 2.0 * eps) * x


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x-np.expand_dims(np.max(x, axis=-1), -1))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)



def get_initial_embedding_target(mob):
    #mob multi-agent observations
    l = mob.shape[0]
    targets = [[mob[i][j][8:11] for j in range(2)] for i in range(l)]
    return np.array(targets)

def cross_entropy(p, q):
    q = np.maximum(q, 1e-20)
    return np.sum(-p*np.log2(q))

def square_loss(p, q):
    return np.sum(np.square(p-q))

def compute_communication_reward(news, ob, b, dis_type):

    # We have the belief of p1 and p2 simultaneously
    distance_fn = get_distance_fn(dis_type)
    rewards = []
    initial_embeddings = get_initial_embedding_target(ob)

    news = np.append(news, 1)

    # compute kl divergence
    i = 0
    for t in range(news.shape[0] - 1):
        if news[t+1]:
            rewards.append([0.0, 0.0])
        else:
            # TODO use another distances.
            r1 = distance_fn(initial_embeddings[t][0], b[t][1]) - distance_fn(initial_embeddings[t][0], b[t+1][1])
            r2 = distance_fn(initial_embeddings[t][1], b[t][0]) - distance_fn(initial_embeddings[t][1], b[t+1][0])
            rewards.append([r1, r2])
    return np.asarray(rewards)


def h_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def get_distance_fn(dis_type):
    if dis_type.lower() == 'kl':
        return ss.entropy
    elif dis_type.lower() == 'hl':
        return h_distance
    elif dis_type.lower() == 'ce':
        return cross_entropy
    elif dis_type.lower() == 'sl':
        return square_loss
    else:
        raise Exception("Unrecognized distance type")


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = [action_space.sample() for action_space in env.action_space] # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = [0, 0] # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)]) # 3D array
    """ - FIXED: Two Agents - """
    rews = np.zeros((horizon, 2), 'float32') # 2D array
    vpreds = np.zeros((horizon, 2), 'float32') # 2D array
    news = np.zeros(horizon, 'int32') # shared
    acs = np.array([ac for _ in range(horizon)]) # 2D array

    print(obs.shape,"observation")
    print(rews.shape, "rews")
    print(acs.shape, "acs")

    while True:
        ac0, vpred0 = pi.act(stochastic, ob[0])
        ac1, vpred1 = pi.act(stochastic, ob[1])
        ac = [ac0, ac1]
        vpred = [vpred0, vpred1]

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "nextvpred": np.array([vpred0, vpred1]) * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret[0] += rew[0]
        cur_ep_ret[1] += rew[1]

        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = [0, 0]
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    # I think is has been simplified
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.concatenate([seg["vpred"], seg["nextvpred"].reshape(1, -1)], axis=0)
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty((T, 2), 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)
