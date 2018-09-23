import tensorflow as tf
from net_code.mlp_policy import MlpPolicy
import numpy as np
from utils import print_args
from net_code.training_utils import loadFromFlat
import os
from WrapperEnv import WrapperEnv
import time


def run_episode(env, pi, time_interval=0.1, stochastic=False, render=False):
    ob = env.reset()

    ep_len = 0  # len of current episode
    ep_rew = 0
    vs = []

    while True:
        ac0, _, _, v1 = pi.act(stochastic, ob[0])
        ac1, _, _, v2 = pi.act(stochastic, ob[1])
        ac = [ac0, ac1]
        vs.append([v1, v2])
        if time_interval:
            time.sleep(time_interval)
        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()
        ep_rew += np.sum(rew)
        ep_len += 1
        if new:
            return ep_len, ep_rew, vs

def learn(config):
    version ='v' + str(config['version'])
    model_to_load_path = os.path.join(config['model_dir'], version)


    env = WrapperEnv(config["scenario_name"], config["max_env_steps"], benchmark=True)
    config["dim_obs"] = env.dim_obs
    config["dim_belief"] = env.dim_belief  # fill the sizes

    # build model
    tf.reset_default_graph()
    pi = MlpPolicy(config)

    print("The Network Can be Created successfully")


    if config['use_GPU']:
        device_config = tf.ConfigProto()
        device_config.gpu_options.allow_growth = True
    else:
        device_config = tf.ConfigProto()

    with tf.Session(config=device_config) as sess:
        sess.run(tf.global_variables_initializer())  # initialized
        model_to_load_path = os.path.join(model_to_load_path, config['model_name'] + '.p')
        if os.path.exists(model_to_load_path):
            loadFromFlat(pi.get_trainable_variables(), model_to_load_path)
            print("Policy Net parameters loaded from %s" % model_to_load_path)
        # else:
        #     raise Exception("Model doesn't exist")

        for i in range(config["num_episodes"]):
            len, ret, v = run_episode(env, pi, config["time_interval"], stochastic=False, render=config["render"])
            print("episode %i length %i return %.3f value %.2f" % (i, len, ret, np.mean(v)))

def main():
    import argparse
    # global hyper parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_GPU', help='if use gpu', default=True)
    parser.add_argument('--render', help='render the environment', default=True)

    parser.add_argument('--hid-layers-sizes', help='the sizes of each hidden layer', default=[64, 64])
    parser.add_argument('--model_dir', help='model loading directory', type=str, default='../../mamodel_p/')
    parser.add_argument('--model_name', help='model name for initial model', type=str, default='pmodel_18')
    parser.add_argument('--scenario-name', '-sn', help='scenario name', type=str, default='simple_body_language_pt')
    parser.add_argument('--max-env-steps', help='maximum steps in the env', type=int, default=25)
    parser.add_argument('--time-interval', help="time interval between two steps", type=float, default=0.2)
    parser.add_argument('--num-episodes', help='number of episodes', type=int, default=10)

    args = parser.parse_args()
    print_args(args)
    args = vars(parser.parse_args())
    learn(args)


if __name__ == '__main__':
    main()
