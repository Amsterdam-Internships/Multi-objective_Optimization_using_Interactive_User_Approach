import os
import numpy as np
import sys
sys.path.append('/src/gp_preference')
sys.path.append('../gp_preference')
from projThompson.decision_maker import DecisionMaker
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import guts
import utilities
import argparse
import bandits

RESULTS_DIR = './results'

if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


def get_filepath(*kwargs):
    """
    Just dump all (hyper)parameters in the arguments here that change between experiments.
    This method will check if we already computed the results and return the filename,
    and otherwise return a new filename at which the results should be stored.
    :param kwargs:
    :return:
    """

    # turn the settings into a string
    filepath = ''
    for i in kwargs:
        filepath += str(i)

    # replace weird symbols with -
    filepath = filepath.replace('.', '-') + '_sanity'

    return os.path.join(RESULTS_DIR, filepath)


# -----------------
# --- settings ----

parser = argparse.ArgumentParser(description='GUTS experiments')
parser.add_argument('--num_objectives', type=int, default=2)
parser.add_argument('--poly_degree', type=int, default=3)
parser.add_argument('--num_iter', type=int, default=1000)
parser.add_argument('--sigth', type=float, default=0.01)
parser.add_argument('--var_std', type=float, default=0.05)
parser.add_argument('--print_logs', type=bool, default=True)
parser.add_argument('--cool', type=int, default=5)
args = parser.parse_args()

num_objectives = args.num_objectives
poly_degree = args.poly_degree
num_iter = args.num_iter
sigth = args.sigth
print_logs = args.print_logs
cool = args.cool
var_std = args.var_std

for num_seeds in range(1, 21):
    plt.figure(1, figsize=(4, 3))
    plt.figure(2, figsize=(4, 3))
    for utility_std in [0.1, 0.5, 1.0]:
        for ground_truth in [False, True]:
            for sig in [True]:  # significance
                for add_virtual_comp in [False]:  # virtual comparisons

                    if ground_truth:
                        if not (sig and not add_virtual_comp and utility_std == 1):
                            continue

                    regret_list = []
                    pull_count_list = []
                    num_questions_list = []

                    for seed in range(num_seeds):
                        seed += 1234

                        if not ground_truth:
                            filepath = get_filepath(num_objectives, poly_degree, utility_std, num_iter, sigth, var_std, cool, sig, add_virtual_comp, seed, ground_truth)
                        else:
                            filepath = get_filepath(num_objectives, poly_degree, num_iter, var_std, seed, ground_truth)
                        if os.path.exists(filepath + '.npz'):
                            regret, pull_counts, num_questions = np.load(filepath+'.npz')['arr_0']
                        else:
                            # pick a (random) utility function
                            utility_function = utilities.random_polynomial_of_order_n(poly_degree, num_objectives, 0.3, 1.2, seed=seed)

                            # initialise decision maker
                            decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std,
                                                           add_virtual_comp=add_virtual_comp)

                            # initialise ground truth bandit
                            mabby = bandits.GaussianBandit(no_obj=num_objectives, no_arms=20, varStd=var_std, predefined_seed=seed)

                            # run gp-ITS
                            regret, pull_counts, num_questions = guts.gp_utility_thompson_sampling(mabby, decision_maker, num_iter,
                                                                                                   print_logs=print_logs, cool=cool,
                                                                                                   initcool=0, sig_test=sig,
                                                                                                   sig_threshold=sigth,
                                                                                                   ground_truth=ground_truth)

                            # save the results
                            np.savez(filepath, [regret, pull_counts, num_questions])

                        regret_list.append(np.cumsum(regret))
                        pull_count_list.append(np.cumsum(pull_counts))
                        num_questions_list.append(np.cumsum(num_questions))

                    if ground_truth:
                        label = 'cheat TS'
                        style = '-.'
                        color ='green'
                    else:
                        label = 'GUTS utl_std={}'.format(utility_std)
                        if utility_std == 0.1:
                            style = ':'
                        elif utility_std == 0.5:
                            style = '--'
                        elif utility_std == 1.0:
                            style = '-'
                        color = 'black'

                    plt.figure(1)
                    p_ts = plt.plot(range(num_iter), np.average(regret_list, axis=0), style, color=color, label=label)
                    plt.xlabel('time', fontsize=15)
                    plt.ylabel('regret', fontsize=15)
                    plt.legend(fontsize=10)

                    if not ground_truth:
                        plt.figure(2)
                        p_qs = plt.plot(range(num_iter), np.average(num_questions_list, axis=0), style, color=color, label=label)
                        plt.xlabel('time', fontsize=15)
                        plt.ylabel('no. queries', fontsize=15)

    # plot results and save figure
    plt.figure(1)
    plt.tight_layout()
    filepath_figure = get_filepath(num_objectives, poly_degree, sigth, var_std) + '_regret'
    plt.savefig(filepath_figure)
    plt.close()

    plt.figure(2)
    plt.tight_layout()
    filepath_figure = get_filepath(num_objectives, poly_degree, sigth, var_std) + '_queries'
    plt.savefig(filepath_figure)
    plt.close()

