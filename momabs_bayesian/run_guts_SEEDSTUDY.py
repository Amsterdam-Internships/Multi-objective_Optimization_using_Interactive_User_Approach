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

RESULTS_DIR = '/data/results'

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
parser.add_argument('--num_iter', type=int, default=500)
parser.add_argument('--sigth', type=float, default=0.01)
parser.add_argument('--var_std', type=float, default=0.05)
parser.add_argument('--print_logs', type=bool, default=True)
parser.add_argument('--cool', type=int, default=0)
args = parser.parse_args()

num_objectives = args.num_objectives
poly_degree = args.poly_degree
num_iter = args.num_iter
sigth = args.sigth
print_logs = args.print_logs
cool = args.cool
var_std = args.var_std

ground_truth = False
sig = True
add_virtual_comp = False
utility_std = 1.0

plt.figure(figsize=(10, 5))

regret_list = []
pull_count_list = []
num_questions_list = []


for i in range(20):
    seed = 1234 + i

    filepath = get_filepath(num_objectives, poly_degree, utility_std, num_iter, sigth, var_std, cool, sig, add_virtual_comp, seed, ground_truth)
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

    plt.subplot(1, 2, 1)
    plt.title('individual run')
    p_ts = plt.plot(range(num_iter), np.cumsum(regret, axis=0), label=seed)
    plt.xlabel('time')
    plt.ylabel('regret')

    plt.subplot(1, 2, 2)
    plt.title('average until here')
    p_qs = plt.plot(range(num_iter), np.average(regret_list, axis=0), label=seed)
    plt.xlabel('time')
    plt.ylabel('no. questions')

# plot results and save figure
plt.legend()
plt.tight_layout()
filepath_figure = get_filepath(num_objectives, poly_degree, sigth, var_std) + '_special-seed' + str(seed)
plt.savefig(filepath_figure)
plt.close()
