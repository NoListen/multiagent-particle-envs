import tensorflow as tf
from net_code.training_utils import traj_segment_generator, add_vtarg_and_adv, zipsame
from net_code.mlp_policy import MlpPolicy
from dataset import Dataset
import numpy as np
from utils import print_args
from net_code.training_utils import saveToFlat
import os
from WrapperEnv import WrapperEnv
import net_code.tf_utils as U

# TODO Two version
# Reinforce and PPO

def learn(config):
    version = 'v0'
    model_to_save_path = os.path.join(config['model_dir_p'], version)
    if not os.path.exists(model_to_save_path):
        os.mkdir(model_to_save_path)

    optim_epochs = config['optim_epochs']
    optim_batchsize = config['optim_batchsize']

    gamma = config['reward_discount']  # advantage estimation
    lam = config['lam']

    max_iters = config['num_iters']  # time constraint
    horizon = config['episodes_per_iter'] * config["max_env_steps"]
    horizon_test = config['episodes_per_iter'] * config["max_env_steps"]*2

    lr_start = config['lr_start']
    lr_decay_iters = config['lr_decay_iters']
    lr_decay_rate = config['lr_decay_rate']

    # Done
    env = WrapperEnv(config["scenario_name"], config["max_env_steps"], benchmark=True, pretrain=True)
    config["dim_obs"] = env.dim_obs
    config["dim_belief"] = env.dim_belief  # fill the sizes
    clip_param = config['clip_param']
    ppo = config['ppo']

    # build model
    tf.reset_default_graph()
    pi = MlpPolicy(config)
    var_list = pi.get_variables()
    if ppo:
        oldpi = MlpPolicy(config, name='old_pi')

    print("The Network Can be Created successfully")

    # setup inputs
    tf_adv = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_adv')  # Empirical return
    tf_ret = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_ret')  # Empirical return
    tf_ac = pi.pdtype.sample_placeholder([None], name='tf_ac')  # action

    # setup losses
    if ppo:
        ratio = tf.exp(pi.pd.logp(tf_ac) - oldpi.pd.logp(tf_ac)) # pnew / pold
        surr1 = ratio * tf_adv # surrogate from conservative policy iteration
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * tf_adv #
        policy_loss = - U.mean(tf.minimum(surr1, surr2))
    else:
        policy_loss = -tf.reduce_mean(tf.multiply(pi.pd.logp(tf_ac), tf_adv))
    value_loss = tf.reduce_mean(tf.square(pi.vpred - tf_ret))
    # TODO you may want some weights here
    total_loss = policy_loss + value_loss*config["valueloss_weight"]

    policy_summary = tf.summary.scalar('policy loss', total_loss)

    # learning rate decay
    global_step_p = tf.Variable(0, trainable=False, name='step_p')
    lr_p = tf.train.exponential_decay(lr_start, global_step_p, lr_decay_iters, lr_decay_rate)
    if ppo:
        assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    # setup optimizer
    opt_for_policy = tf.train.AdamOptimizer(learning_rate=lr_p) \
        .minimize(total_loss, name='opt_p', global_step=global_step_p, var_list=var_list)

    iters_so_far = 0
    summary_op_policy = tf.summary.merge([policy_summary])

    if config['use_GPU']:
        device_config = tf.ConfigProto()
        device_config.gpu_options.allow_growth = True
    else:
        device_config = tf.ConfigProto()

    """The env has fixed length and won't terminate when one get to the target.
    Please Ensure [horizon % max_env_steps] by hand 
    For convenience that each episode is of fixed length
    By this way, the data can be used more efficiently."""
    seg_gen = traj_segment_generator(pi, env, horizon, stochastic=True)
    seg_gen_test = traj_segment_generator(pi, env, horizon_test, stochastic=False)

    best_reward = -np.inf
    best_iter = 0
    with tf.Session(config=device_config) as sess:
        sess.run(tf.global_variables_initializer())  # initialized
        while True:
            if iters_so_far >= max_iters:
                break
            print("********** Iteration %i ************" % iters_so_far)
            # get a set of trajectories for current iteration based on current network
            seg = seg_gen.__next__()

            # starting_ob = get_initial_obs(seg["new"], seg["ob"])
            optim_batchsize = optim_batchsize  # or ob.shape[0]  # number of obs is less than batchsize

            """Check about the shape!!"""
            add_vtarg_and_adv(seg, gamma, lam)

            """The useful data for training."""
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            atarg = (atarg - atarg.mean()) / atarg.std()
            # reshape
            print(np.mean(seg["vpred"]), "Value prediction", np.mean(seg["tdlamret"]))

            ob = ob.reshape(-1, ob.shape[-1])
            ac = ac.reshape(-1)
            atarg = atarg.reshape(-1)
            tdlamret = tdlamret.reshape(-1)

            # print(ob.shape, "Now the observation has shape")
            # print(ac.shape, "None the ac has shape")
            # print(atarg.shape, "None the atarg has shape")
            # print(tdlamret.shape, "None the tdlamret has shape")

            # initialise a Dataset instance
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
            if ppo:
                assign_old_eq_new()
            # training policy net
            total_losses = []  # list of tuples, each of which gives the loss for a minibatch
            p_losses = []
            v_losses = []
            for _ in range(optim_epochs):
                for batch in d.iterate_once(optim_batchsize):
                    if ppo:
                        _, batch_loss, p_loss, v_loss, summary_str = sess.run([opt_for_policy,
                                                                               total_loss,
                                                                               policy_loss,
                                                                               value_loss,
                                                                               summary_op_policy],
                                                                              feed_dict={pi.ob: batch["ob"],
                                                                                         pi.stochastic: True,
                                                                                         oldpi.ob: batch["ob"],
                                                                                         oldpi.stochastic: True,
                                                                                         tf_ac: batch['ac'],
                                                                                         tf_ret: batch['vtarg'],
                                                                                         tf_adv: batch["atarg"]})
                    else:
                        _, batch_loss, p_loss, v_loss, summary_str = sess.run([opt_for_policy,
                                                                               total_loss,
                                                                               policy_loss,
                                                                               value_loss,
                                                                               summary_op_policy],
                                                                               feed_dict={pi.ob: batch["ob"],
                                                                               pi.stochastic: True,
                                                                               tf_ac: batch['ac'],
                                                                               tf_ret: batch['vtarg'],
                                                                               tf_adv: batch["atarg"]})
                    total_losses.append(batch_loss)
                    p_losses.append(p_loss)
                    v_losses.append(v_loss)
            print("Mean loss:\t %.4f Policy loss:\t %.4f Value loss:\t %.4f" % (np.mean(total_losses),
                                                                                np.mean(p_losses),
                                                                                np.mean(v_losses)))
            seg = seg_gen_test.__next__()
            test_reward = np.mean(seg['rew'])
            print("test mean reward in current iteration: %.4f" % test_reward)
            if test_reward > best_reward:
                best_reward = test_reward
                best_iter = iters_so_far
                file_name = os.path.join(model_to_save_path, "model_best.p")
                saveToFlat(pi.get_trainable_variables(), file_name)
                print('current best iteration is %s' % best_iter)

            if iters_so_far % 2 == 0:
                # the global model is not stored yer
                file_name = os.path.join(model_to_save_path, "model_%i.p" % iters_so_far)
                saveToFlat(pi.get_trainable_variables(), file_name)
                print('saved %s' % file_name)
                # saver.save(sess, model_path + 'model_%i.checkpoint' % iters_so_far)
            iters_so_far += 1
        print('The final best iteration is %i and test reward is %.f' % (best_iter, best_reward))


def main():
    import argparse
    # global hyper parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_GPU', help='if use gpu', default=True)
    parser.add_argument('--hid-layers-sizes', help='the sizes of each hidden layer', default=[64, 64])
    parser.add_argument('--dis_type', '-dt', help='type of distance to use', default='kl')
    parser.add_argument('--model_dir', help='model saving directory', type=str, default='../../mamodel_p/')
    parser.add_argument('--dir_logs', '-dl', help='logs saving directory', type=str, default='../../logs/')

    parser.add_argument('--scenario-name', '-sn', help='scenario name', type=str, default='simple3')
    parser.add_argument('--max-env-steps', '-mss', help='maximum steps in the env', type=int, default=25)
    # online learning hyper parameters
    parser.add_argument('--optim_epochs', '-oe', help='optimising epochs', type=int, default=5)
    parser.add_argument('--optim_batchsize', '-ob', help='optimising batch size', type=int, default=1024)
    parser.add_argument('--reward_discount', '-rd', help='reward discount factor', type=float, default=0.9)
    parser.add_argument('--lam', '-lam', help='reward discount factor', type=float, default=0.95)
    parser.add_argument('--num_iters', '-ni', help='number of iterations of training', type=int, default=300)
    parser.add_argument('--episodes-per-iter', '-epi', help='number of episodes per actorbatch', type=int, default=500)
    parser.add_argument('--lr_start', default=0.001)
    parser.add_argument('--valueloss-weight', default=1.0)
    parser.add_argument('--lr_decay_rate', default=0.95)
    parser.add_argument('--lr_decay_iters', default=400)
    parser.add_argument('--ppo', '-ppo', default=True)
    parser.add_argument('--clip_param', '-cp', default=0.2, type=float)
    args = parser.parse_args()
    print_args(args)
    args = vars(parser.parse_args())
    learn(args)


if __name__ == '__main__':
    main()
