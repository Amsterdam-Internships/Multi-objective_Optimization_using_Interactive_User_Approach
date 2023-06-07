#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:37:57 2017

@author: Diederik M. Roijers (Vrije Universiteit Brussel)
"""

import bandits
import numpy as np
import sys
sys.path.append('../gp_preference')
sys.path.append('../pymodem')
from projThompson.decision_maker import DecisionMaker
import matplotlib.pyplot as plt
import time
import utilities
from adt17Algos import interactive_thompson_sampling, LinearDecisionMaker


def gp_utility_thompson_sampling(bandit, decision_maker, num_iter, print_logs=False,
                                 cool=0, initcool=0, sig_test=False, sig_threshold=0.05, 
                                 ground_truth=False):
    """
    Run Iterative Thompson Sampling with GP preferences and a Multi-Armed Bandit
    :param bandit:          instance of bandits.BernoulliBandit 
    :param decision_maker:  instance of gp_preference.thompson.DecisionMaker
    :param num_iter: 
    :param print_logs: 
    :param cool: 
    :param initcool: 
    :return: 
    """
    # keep track of the bandit stats (like arm pulls)
    bandit_stats = bandits.GaussianBanditStats(bandit)

    # record regret and number of questions per time step
    regret = np.zeros(num_iter)
    questions = np.zeros(num_iter)

    #print(bandit.arms)

    gt_vec_ = list(map(decision_maker.true_utility, bandit.arms))
    gt_vec = (gt_vec_ - min(gt_vec_)) / (max(gt_vec_) - min(gt_vec_))
    true_optimum = max(gt_vec)
    print(true_optimum)
    regret_vec = [true_optimum - x for x in gt_vec]
    # weight_distances = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    print(regret_vec)
    cooldown = initcool

    for i in range(num_iter):

        # weights, w_fit, H_fit = decision_maker.current_map()
        start_time = time.time()

        # get two samples from the bandit (for all arms)
        vs1 = bandit_stats.ts_sample()
        vs2 = bandit_stats.ts_sample()

        # reduce to undominated set
        ps1 = utilities.pareto_front(vs1)
        ps2 = utilities.pareto_front(vs2)

        # sample the utility from the user GP at the undominated arms
        
        if ground_truth:
            us1 = list(map(decision_maker.true_utility,vs1))
            us2 = list(map(decision_maker.true_utility,vs2))
        else: 
            us1 = decision_maker.sample(ps1)
            us2 = decision_maker.sample(ps2)

        # find the arm index that maximises user utility according to these samples
        am_p = np.argmax(us1)
        au_p = np.argmax(us2)

        # get the index of the arm with the highest utility in the undominated set
        
        if ground_truth:
            am = am_p
            au = au_p
        else:
            am = utilities.find_vector(ps1[am_p], vs1)
            au = utilities.find_vector(ps2[au_p], vs2)

        # w.l.o.g., pull arm 1 to get a reward
        reward = bandit.pull(am)
        bandit_stats.enter_data(am, reward)

        # get regret for pulling that arm
        regret[i] = regret_vec[am]

        # if the arms (am and au) are different, ask the user for a comparison
        if am != au and cooldown <= 0 and len(utilities.pareto_front([vs1[am], vs2[au]])) == 2 and not ground_truth:
            # em = stats.current_estimate(am)
            # eu = stats.current_estimate(au)  
            if (not sig_test):
                em = vs1[am]
                eu = vs2[au]
                # let the user compare the two vectors
                decision_maker.noisy_compare(em, eu)
                questions[i] = 1
                cooldown = cool
                sig_str = " /"
            else:
                # only ask the user if the datasets for the two arms
                # are significantly different (hotellings T^2 test)
                if bandit_stats.significance_test_arms(am, au, sig_threshold):
                    em = vs1[am]
                    eu = vs2[au]
                    # let the user compare the two vectors
                    decision_maker.noisy_compare(em, eu)
                    questions[i] = 1
                    cooldown = cool
                    sig_str = " significant ("+str(sum(questions))+")"
                else:
                    cooldown = cooldown - 1
                    sig_str = " not sign."
        else:
            # em = bandit_stats.current_estimate(am)
            cooldown = cooldown - 1
            sig_str = " #"

        if print_logs:
            print(i, ',', round(time.time()-start_time, 5), ',', 'regret', np.round(np.sum(regret), 2), am != au, ',', am, ' vs ', au, 'cd:', cooldown, ',', sig_str)

    return regret, list(bandit_stats.counts), questions



def experimentExmple5(horizon, repeat, seed_=42):
    figx = 3.0
    figy = 2.1
    dpix = 120

    the_seed = seed_
    num_objectives = 2
    utility_std = 0.01 #was 0.01
    num_iter = horizon
    print_logs = False
    initcool = 0
    sig=True
    sigth=0.01
    arm_variance=0.01
    temp_linear_prior=False
    
    
    #Polynomial for the example problem
    terms = [[(0,1),(1,1)]]
    coeffs= [6.25]
    utility_function = utilities.lambda_polynomial(terms, coeffs)
    
    # initialise ground truth bandit
    mabby = bandits.GaussianBandit(2, 20, varStd=arm_variance, predefined_seed=the_seed)
    mabby.redef_self_5arm_example()
    
    guts0 = [0] * horizon
    guts1 = [0] * horizon
    guts2 = [0] * horizon
    guts3 = [0] * horizon
    ts_cheat = [0] * horizon
    its = [0] * horizon
    qguts0 = [0] * horizon
    qguts1 = [0] * horizon
    qguts2 = [0] * horizon
    qguts3 = [0] * horizon
    qts_cheat = [0] * horizon
    qits = [0] * horizon
    f = open('datalog.py', 'a')
    for i in range(repeat):
        gt=False
        seed = the_seed+i*7
        # initialise decision maker
        add_virtual_comp = True 
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        # run vanilla ITS:
        gt_vec_ = list(map(decision_maker.true_utility, mabby.arms))
        gt_vec = (gt_vec_ - min(gt_vec_)) / (max(gt_vec_) - min(gt_vec_))
        true_optimum = max(gt_vec)
        print(true_optimum)
        regret_vec = [true_optimum - x for x in gt_vec]
        linear_dm = LinearDecisionMaker(num_objectives, utility_std, defer_comp_to=decision_maker)
        regret_its, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, linear_dm, num_iter, prespecified_regret_vector=regret_vec)
    
        its = [its[cnt] + regret_its[cnt] for cnt in range(len(its))]
        qits = [qits[cnt] + qumap[cnt] for cnt in range(len(qits))]
        f.write("its_" + str(i) + " = " + str(its) + "\n")
        f.write("qits_" + str(i) + " = " + str(qits) + "\n")
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = True
        cool = 0    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
    
        regret_0, pull_counts_0, num_questions_0 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg0 = np.cumsum(regret_0)
        cumq0 = np.cumsum(num_questions_0)
        guts0 = [guts0[cnt] + cumreg0[cnt] for cnt in range(len(guts0))]
        qguts0 = [qguts0[cnt] + cumq0[cnt] for cnt in range(len(qguts0))]
        f.write("guts0_" + str(i) + " = " + str(guts0) + "\n")
        f.write("qguts0_" + str(i) + " = " + str(qguts0) + "\n")
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = True
        cool = 1    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
    
        regret_1, pull_counts_1, num_questions_1 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg1 = np.cumsum(regret_1)
        cumq1 = np.cumsum(num_questions_1)
        guts1 = [guts1[cnt] + cumreg1[cnt] for cnt in range(len(guts1))]
        qguts1 = [qguts1[cnt] + cumq1[cnt] for cnt in range(len(qguts1))]
        f.write("guts1_" + str(i) + " = " + str(guts1) + "\n")
        f.write("qguts1_" + str(i) + " = " + str(qguts1) + "\n")
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = True
        cool = 2    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_2, pull_counts_2, num_questions_2 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg2 = np.cumsum(regret_2)
        cumq2 = np.cumsum(num_questions_2)
        guts2 = [guts2[cnt] + cumreg2[cnt] for cnt in range(len(guts2))]
        qguts2 = [qguts2[cnt] + cumq2[cnt] for cnt in range(len(qguts2))]
        f.write("guts2_" + str(i) + " = " + str(guts2) + "\n")
        f.write("qguts2_" + str(i) + " = " + str(qguts2) + "\n")
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = True
        cool = 3    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_3, pull_counts_3, num_questions_3 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg3 = np.cumsum(regret_3)
        cumq3 = np.cumsum(num_questions_3)
        guts3 = [guts3[cnt] + cumreg3[cnt] for cnt in range(len(guts3))]
        qguts3 = [qguts3[cnt] + cumq3[cnt] for cnt in range(len(qguts3))]
        f.write("guts3_" + str(i) + " = " + str(guts3) + "\n")
        f.write("qguts3_" + str(i) + " = " + str(qguts3) + "\n")

        # run cheat TS
        gt=True     
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_ch, pull_counts_ch, num_questions_ch = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumregch = np.cumsum(regret_ch)
        cumqch = np.cumsum(num_questions_ch)
        ts_cheat = [ts_cheat[cnt] + cumregch[cnt] for cnt in range(len(ts_cheat))]
        qts_cheat = [qts_cheat[cnt] + cumqch[cnt] for cnt in range(len(qts_cheat))]
        f.write("gutsc_" + str(i) + " = " + str(ts_cheat) + "\n")
        f.write("qgutsc_" + str(i) + " = " + str(qts_cheat) + "\n")

        print("iter. " + str(i)+ " done")
        
    f.close()
    guts0 = [x / repeat for x in guts0]
    guts1 = [x / repeat for x in guts1]
    guts2 = [x / repeat for x in guts2]
    guts3 = [x / repeat for x in guts3]
    ts_cheat = [x / repeat for x in ts_cheat]
    its = [x / repeat for x in its]
    qguts0 = [x / repeat for x in qguts0]
    qguts1 = [x / repeat for x in qguts1]
    qguts2 = [x / repeat for x in qguts2]
    qguts3 = [x / repeat for x in qguts3]
    qts_cheat = [x / repeat for x in qts_cheat]
    qits = [x / repeat for x in qits]
    
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), its, color='blue')
    p_ts = plt.plot(range(horizon), guts2, color='black')
    p_naive = plt.plot(range(horizon), guts1, '--', color='black')
    p_tsch = plt.plot(range(horizon), guts0, ':', color='black')
    p_ch = plt.plot(range(horizon), guts3, color='gray')
    p_chc = plt.plot(range(horizon), ts_cheat, '-.', color='green')
    # p_random = plt.plot(range(n),[0.340195531946*x for x in range(n)], color='red')
    # p_random = plt.plot(range(horizon),[0.21329954363*x for x in range(horizon)], color='red')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.legend((p_us[0], p_tsch[0], p_naive[0], p_ts[0], p_ch[0], p_chc[0]),
                     ('ITS', 'GUTS cd=0', 'GUTS cd=1', 'GUTS cd=2', 'GUTS cd=3', 'cheat TS'),
                     loc='upper left', fontsize=6)
    # leg.get_frame().set_alpha(0.5)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_g0 = plt.plot(range(horizon), qguts0, ':', color='black')
    p_g1 = plt.plot(range(horizon), qguts1, '--', color='black')
    p_g2 = plt.plot(range(horizon), qguts2, color='black')
    p_g3 = plt.plot(range(horizon), qguts3, color='gray')
    p_its = plt.plot(range(horizon), qits, color='blue')
    #p_cheat = plt.plot(range(horizon), qts_cheat, color='green')
    #plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'),
    #           loc='upper left')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. queries')
    plt.show()


def experimentExample5significance(horizon, repeat, seed_=42):
    figx = 3.0
    figy = 2.1
    dpix = 120

    the_seed = seed_
    num_objectives = 2
    utility_std = 0.01 #was 0.01
    num_iter = horizon
    print_logs = False
    initcool = 0
    sig=True
    sigth=0.01
    
    #Polynomial for the example problem
    terms = [[(0,1),(1,1)]]
    coeffs= [6.25]
    utility_function = utilities.lambda_polynomial(terms, coeffs)
    
    # initialise ground truth bandit
    mabby = bandits.GaussianBandit(2, 20, varStd=0.01, predefined_seed=the_seed)
    mabby.redef_self_5arm_example()
    
    guts0 = [0] * horizon
    #guts1 = [0] * horizon
    #guts2 = [0] * horizon
    guts3 = [0] * horizon
    ts_cheat = [0] * horizon
    its = [0] * horizon
    qguts0 = [0] * horizon
    #qguts1 = [0] * horizon
    #qguts2 = [0] * horizon
    qguts3 = [0] * horizon
    qts_cheat = [0] * horizon
    qits = [0] * horizon
    f = open('datalog.py', 'a')
    for i in range(repeat):
        gt=False
        seed = the_seed+i*7
        # initialise decision maker
        add_virtual_comp = True 
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=False, add_virtual_comp=False)
        # run vanilla ITS:
        gt_vec_ = list(map(decision_maker.true_utility, mabby.arms))
        gt_vec = (gt_vec_ - min(gt_vec_)) / (max(gt_vec_) - min(gt_vec_))
        true_optimum = max(gt_vec)
        print(true_optimum)
        regret_vec = [true_optimum - x for x in gt_vec]
        linear_dm = LinearDecisionMaker(num_objectives, utility_std, defer_comp_to=decision_maker)
        regret_its, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, linear_dm, num_iter, prespecified_regret_vector=regret_vec)
    
        its = [its[cnt] + regret_its[cnt] for cnt in range(len(its))]
        qits = [qits[cnt] + qumap[cnt] for cnt in range(len(qits))]
        f.write("its_" + str(i) + " = " + str(its) + "\n")
        f.write("qits_" + str(i) + " = " + str(qits) + "\n")
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = False
        cool = 0    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=False, add_virtual_comp=False)
        print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
    
        regret_0, pull_counts_0, num_questions_0 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg0 = np.cumsum(regret_0)
        cumq0 = np.cumsum(num_questions_0)
        guts0 = [guts0[cnt] + cumreg0[cnt] for cnt in range(len(guts0))]
        qguts0 = [qguts0[cnt] + cumq0[cnt] for cnt in range(len(qguts0))]
        f.write("guts0_" + str(i) + " = " + str(guts0) + "\n")
        f.write("qguts0_" + str(i) + " = " + str(qguts0) + "\n")
                
        # run gp-ITS
        add_virtual_comp = False 
        sig = True
        cool = 0    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=False, add_virtual_comp=False)
        regret_3, pull_counts_3, num_questions_3 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg3 = np.cumsum(regret_3)
        cumq3 = np.cumsum(num_questions_3)
        guts3 = [guts3[cnt] + cumreg3[cnt] for cnt in range(len(guts3))]
        qguts3 = [qguts3[cnt] + cumq3[cnt] for cnt in range(len(qguts3))]
        f.write("guts3_" + str(i) + " = " + str(guts3) + "\n")
        f.write("qguts3_" + str(i) + " = " + str(qguts3) + "\n")

        # run cheat TS
        gt=True     
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=False, add_virtual_comp=False)
        regret_ch, pull_counts_ch, num_questions_ch = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumregch = np.cumsum(regret_ch)
        cumqch = np.cumsum(num_questions_ch)
        ts_cheat = [ts_cheat[cnt] + cumregch[cnt] for cnt in range(len(ts_cheat))]
        qts_cheat = [qts_cheat[cnt] + cumqch[cnt] for cnt in range(len(qts_cheat))]
        f.write("gutsc_" + str(i) + " = " + str(ts_cheat) + "\n")
        f.write("qgutsc_" + str(i) + " = " + str(qts_cheat) + "\n")

        print("iter. " + str(i)+ " done")
        
    f.close()
    guts0 = [x / repeat for x in guts0]
#    guts1 = [x / repeat for x in guts1]
#    guts2 = [x / repeat for x in guts2]
    guts3 = [x / repeat for x in guts3]
    ts_cheat = [x / repeat for x in ts_cheat]
    its = [x / repeat for x in its]
    qguts0 = [x / repeat for x in qguts0]
#    qguts1 = [x / repeat for x in qguts1]
#    qguts2 = [x / repeat for x in qguts2]
    qguts3 = [x / repeat for x in qguts3]
    qts_cheat = [x / repeat for x in qts_cheat]
    qits = [x / repeat for x in qits]
    
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), its, color='blue')
#    p_ts = plt.plot(range(horizon), guts2, color='black')
#    p_naive = plt.plot(range(horizon), guts1, '--', color='black')
    p_tsch = plt.plot(range(horizon), guts0, ':', color='black')
    p_ch = plt.plot(range(horizon), guts3, color='black')
    p_chc = plt.plot(range(horizon), ts_cheat, '-.', color='green')
    # p_random = plt.plot(range(n),[0.340195531946*x for x in range(n)], color='red')
    # p_random = plt.plot(range(horizon),[0.21329954363*x for x in range(horizon)], color='red')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.legend((p_us[0], p_tsch[0], #p_naive[0], p_ts[0], 
                p_ch[0], p_chc[0]),
                     ('ITS', 'GUTS s0', #'GUTS cd=1', 'GUTS cd=2', 
                      'GUTS s1', 'cheat TS'),
                     loc='upper left', fontsize=6)
    # leg.get_frame().set_alpha(0.5)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_g0 = plt.plot(range(horizon), qguts0, ':', color='black')
    #p_g1 = plt.plot(range(horizon), qguts1, '--', color='black')
    #p_g2 = plt.plot(range(horizon), qguts2, color='black')
    p_g3 = plt.plot(range(horizon), qguts3, color='black')
    p_its = plt.plot(range(horizon), qits, color='blue')
    #p_cheat = plt.plot(range(horizon), qts_cheat, color='green')
    #plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'),
    #           loc='upper left')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. queries')
    plt.show()

    
    
def experimentRandomRandom(horizon, repeat, poly_degree=3, arm_variance=0.05, seed_=42):
    figx = 3.0
    figy = 2.1
    dpix = 120

    the_seed = seed_
    num_objectives = 2
    utility_std = 0.01 #was 0.01
    num_iter = horizon
    print_logs = True
    initcool = 0
    sig=True
    sigth=0.01
    temp_linear_prior=False
    
    #Polynomial for the example problem
    terms = [[(0,1),(1,1)]]
    coeffs= [6.25]
    utility_function = utilities.lambda_polynomial(terms, coeffs)
        
    guts0 = [0] * horizon
    guts1 = [0] * horizon
    guts2 = [0] * horizon
    guts3 = [0] * horizon
    ts_cheat = [0] * horizon
    #its = [0] * horizon
    qguts0 = [0] * horizon
    qguts1 = [0] * horizon
    qguts2 = [0] * horizon
    qguts3 = [0] * horizon
    qts_cheat = [0] * horizon
    #qits = [0] * horizon
    f = open('datalog.py', 'a')
    for i in range(repeat):
        gt=False
        seed = the_seed+i*7
        # initialise ground truth bandit
        mabby = bandits.GaussianBandit(2, 20, varStd=arm_variance, predefined_seed=seed)
        # pick a (random) utility function
        utility_function = utilities.random_polynomial_of_order_n(poly_degree, num_objectives, 0.3, 1.2, seed=seed)
        # initialise decision maker
        add_virtual_comp = True 
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        # run vanilla ITS:
        #gt_vec_ = list(map(decision_maker.true_utility, mabby.arms))
        #gt_vec = (gt_vec_ - min(gt_vec_)) / (max(gt_vec_) - min(gt_vec_))
        #true_optimum = max(gt_vec)
        #print(true_optimum)
        #regret_vec = [true_optimum - x for x in gt_vec]
        #linear_dm = LinearDecisionMaker(num_objectives, utility_std, defer_comp_to=decision_maker)
        #regret_its, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, linear_dm, num_iter, prespecified_regret_vector=regret_vec)
    
        #its = [its[cnt] + regret_its[cnt] for cnt in range(len(its))]
        #qits = [qits[cnt] + qumap[cnt] for cnt in range(len(qits))]
        #f.write("its_" + str(i) + " = " + str(its) + "\n")
        #f.write("qits_" + str(i) + " = " + str(qits) + "\n")
        
        # run gp-ITS
        add_virtual_comp = True 
        sig = True
        cool = 0    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        #print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
    
        regret_0, pull_counts_0, num_questions_0 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg0 = np.cumsum(regret_0)
        cumq0 = np.cumsum(num_questions_0)
        guts0 = [guts0[cnt] + cumreg0[cnt] for cnt in range(len(guts0))]
        qguts0 = [qguts0[cnt] + cumq0[cnt] for cnt in range(len(qguts0))]
        f.write("guts0_" + str(i) + " = " + str(guts0) + "\n")
        f.write("qguts0_" + str(i) + " = " + str(qguts0) + "\n")
        
        # run gp-ITS
        add_virtual_comp = True 
        sig = True
        cool = 5   
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        #print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
    
        regret_1, pull_counts_1, num_questions_1 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg1 = np.cumsum(regret_1)
        cumq1 = np.cumsum(num_questions_1)
        guts1 = [guts1[cnt] + cumreg1[cnt] for cnt in range(len(guts1))]
        qguts1 = [qguts1[cnt] + cumq1[cnt] for cnt in range(len(qguts1))]
        f.write("guts1_" + str(i) + " = " + str(guts1) + "\n")
        f.write("qguts1_" + str(i) + " = " + str(qguts1) + "\n")
        
        # run gp-ITS
        add_virtual_comp = True 
        sig = True
        cool = 10    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_2, pull_counts_2, num_questions_2 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg2 = np.cumsum(regret_2)
        cumq2 = np.cumsum(num_questions_2)
        guts2 = [guts2[cnt] + cumreg2[cnt] for cnt in range(len(guts2))]
        qguts2 = [qguts2[cnt] + cumq2[cnt] for cnt in range(len(qguts2))]
        f.write("guts2_" + str(i) + " = " + str(guts2) + "\n")
        f.write("qguts2_" + str(i) + " = " + str(qguts2) + "\n")
        
        # run gp-ITS
        add_virtual_comp = True 
        sig = True
        cool = 20    
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_3, pull_counts_3, num_questions_3 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg3 = np.cumsum(regret_3)
        cumq3 = np.cumsum(num_questions_3)
        guts3 = [guts3[cnt] + cumreg3[cnt] for cnt in range(len(guts3))]
        qguts3 = [qguts3[cnt] + cumq3[cnt] for cnt in range(len(qguts3))]
        f.write("guts3_" + str(i) + " = " + str(guts3) + "\n")
        f.write("qguts3_" + str(i) + " = " + str(qguts3) + "\n")

        # run cheat TS
        gt=True     
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_ch, pull_counts_ch, num_questions_ch = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumregch = np.cumsum(regret_ch)
        cumqch = np.cumsum(num_questions_ch)
        ts_cheat = [ts_cheat[cnt] + cumregch[cnt] for cnt in range(len(ts_cheat))]
        qts_cheat = [qts_cheat[cnt] + cumqch[cnt] for cnt in range(len(qts_cheat))]
        f.write("gutsc_" + str(i) + " = " + str(ts_cheat) + "\n")
        f.write("qgutsc_" + str(i) + " = " + str(qts_cheat) + "\n")

        print("iter. " + str(i)+ " done")
        
    f.close()
    guts0 = [x / repeat for x in guts0]
    guts1 = [x / repeat for x in guts1]
    guts2 = [x / repeat for x in guts2]
    guts3 = [x / repeat for x in guts3]
    ts_cheat = [x / repeat for x in ts_cheat]
    #its = [x / repeat for x in its]
    qguts0 = [x / repeat for x in qguts0]
    qguts1 = [x / repeat for x in qguts1]
    qguts2 = [x / repeat for x in qguts2]
    qguts3 = [x / repeat for x in qguts3]
    qts_cheat = [x / repeat for x in qts_cheat]
    #qits = [x / repeat for x in qits]
    
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    #p_us = plt.plot(range(horizon), its, color='blue')
    p_ts = plt.plot(range(horizon), guts2, color='black')
    p_naive = plt.plot(range(horizon), guts1, '--', color='black')
    p_tsch = plt.plot(range(horizon), guts0, ':', color='black')
    p_ch = plt.plot(range(horizon), guts3, color='gray')
    p_chc = plt.plot(range(horizon), ts_cheat, '-.', color='green')
    # p_random = plt.plot(range(n),[0.340195531946*x for x in range(n)], color='red')
    # p_random = plt.plot(range(horizon),[0.21329954363*x for x in range(horizon)], color='red')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.legend((p_tsch[0], p_naive[0], p_ts[0], p_ch[0], p_chc[0]),
                     ('GUTS cd=0', 'GUTS cd=5', 'GUTS cd=10', 'GUTS cd=20', 'cheat TS'),
                     loc='upper left', fontsize=6)
    # leg.get_frame().set_alpha(0.5)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_g0 = plt.plot(range(horizon), qguts0, ':', color='black')
    p_g1 = plt.plot(range(horizon), qguts1, '--', color='black')
    p_g2 = plt.plot(range(horizon), qguts2, color='black')
    p_g3 = plt.plot(range(horizon), qguts3, color='gray')
    #p_its = plt.plot(range(horizon), qits, '--', color='blue')
    #p_cheat = plt.plot(range(horizon), qts_cheat, color='green')
    #plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'),
    #           loc='upper left')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. queries')
    plt.show()
    
def experimentVariantsRandom(horizon, repeat, poly_degree=3, arm_variance=0.05, no_objectives=2, cd_=0, seed_=42):
    figx = 3.0
    figy = 2.1
    dpix = 120

    the_seed = seed_
    num_objectives = no_objectives
    utility_std = 0.01 #was 0.01
    num_iter = horizon
    print_logs = True
    initcool = 0
    sig=True
    sigth=0.01
    temp_linear_prior=False
    cool = cd_ 
        
    guts0 = [0] * horizon
    #guts1 = [0] * horizon
    #guts2 = [0] * horizon
    guts3 = [0] * horizon
    ts_cheat = [0] * horizon
    #its = [0] * horizon
    qguts0 = [0] * horizon
    #qguts1 = [0] * horizon
    #qguts2 = [0] * horizon
    qguts3 = [0] * horizon
    qts_cheat = [0] * horizon
    #qits = [0] * horizon
    f = open('datalog.py', 'a')
    for i in range(repeat):
        gt=False
        seed = the_seed+i*7
        # initialise ground truth bandit
        mabby = bandits.GaussianBandit(num_objectives, 20, varStd=arm_variance, predefined_seed=seed)
        # pick a (random) utility function
        utility_function = utilities.random_polynomial_of_order_n(poly_degree, num_objectives, 0.3, 1.2, seed=seed)
        # initialise decision maker
        add_virtual_comp = True 
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        # run vanilla ITS:
        #gt_vec_ = list(map(decision_maker.true_utility, mabby.arms))
        #gt_vec = (gt_vec_ - min(gt_vec_)) / (max(gt_vec_) - min(gt_vec_))
        #true_optimum = max(gt_vec)
        #print(true_optimum)
        #regret_vec = [true_optimum - x for x in gt_vec]
        #linear_dm = LinearDecisionMaker(num_objectives, utility_std, defer_comp_to=decision_maker)
        #regret_its, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, linear_dm, num_iter, prespecified_regret_vector=regret_vec)
    
        #its = [its[cnt] + regret_its[cnt] for cnt in range(len(its))]
        #qits = [qits[cnt] + qumap[cnt] for cnt in range(len(qits))]
        #f.write("its_" + str(i) + " = " + str(its) + "\n")
        #f.write("qits_" + str(i) + " = " + str(qits) + "\n")
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = False   
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        #print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
    
        regret_0, pull_counts_0, num_questions_0 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg0 = np.cumsum(regret_0)
        cumq0 = np.cumsum(num_questions_0)
        guts0 = [guts0[cnt] + cumreg0[cnt] for cnt in range(len(guts0))]
        qguts0 = [qguts0[cnt] + cumq0[cnt] for cnt in range(len(qguts0))]
        f.write("guts0_" + str(i) + " = " + str(guts0) + "\n")
        f.write("qguts0_" + str(i) + " = " + str(qguts0) + "\n")
        
        # run gp-ITS
        #add_virtual_comp = True 
        #sig = False  
        #decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        #print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))
        #
        #regret_1, pull_counts_1, num_questions_1 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
        #                                                              cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
        #                                                              ground_truth=gt)
        #cumreg1 = np.cumsum(regret_1)
        #cumq1 = np.cumsum(num_questions_1)
        #guts1 = [guts1[cnt] + cumreg1[cnt] for cnt in range(len(guts1))]
        #qguts1 = [qguts1[cnt] + cumq1[cnt] for cnt in range(len(qguts1))]
        #f.write("guts1_" + str(i) + " = " + str(guts1) + "\n")
        #f.write("qguts1_" + str(i) + " = " + str(qguts1) + "\n")
        
        # run gp-ITS
        #add_virtual_comp = False 
        #sig = True   
        #decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        #regret_2, pull_counts_2, num_questions_2 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
        #                                                              cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
        #                                                              ground_truth=gt)
        #cumreg2 = np.cumsum(regret_2)
        #cumq2 = np.cumsum(num_questions_2)
        #guts2 = [guts2[cnt] + cumreg2[cnt] for cnt in range(len(guts2))]
        #qguts2 = [qguts2[cnt] + cumq2[cnt] for cnt in range(len(qguts2))]
        #f.write("guts2_" + str(i) + " = " + str(guts2) + "\n")
        #f.write("qguts2_" + str(i) + " = " + str(qguts2) + "\n")
        #
        
        # run gp-ITS
        add_virtual_comp = False 
        sig = True   
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_3, pull_counts_3, num_questions_3 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumreg3 = np.cumsum(regret_3)
        cumq3 = np.cumsum(num_questions_3)
        guts3 = [guts3[cnt] + cumreg3[cnt] for cnt in range(len(guts3))]
        qguts3 = [qguts3[cnt] + cumq3[cnt] for cnt in range(len(qguts3))]
        f.write("guts3_" + str(i) + " = " + str(guts3) + "\n")
        f.write("qguts3_" + str(i) + " = " + str(qguts3) + "\n")

        # run cheat TS
        gt=True     
        decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
        regret_ch, pull_counts_ch, num_questions_ch = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
        cumregch = np.cumsum(regret_ch)
        cumqch = np.cumsum(num_questions_ch)
        ts_cheat = [ts_cheat[cnt] + cumregch[cnt] for cnt in range(len(ts_cheat))]
        qts_cheat = [qts_cheat[cnt] + cumqch[cnt] for cnt in range(len(qts_cheat))]
        f.write("gutsc_" + str(i) + " = " + str(ts_cheat) + "\n")
        f.write("qgutsc_" + str(i) + " = " + str(qts_cheat) + "\n")

        print("iter. " + str(i)+ " done")
        
    f.close()
    guts0 = [x / repeat for x in guts0]
    #guts1 = [x / repeat for x in guts1]
    #guts2 = [x / repeat for x in guts2]
    guts3 = [x / repeat for x in guts3]
    ts_cheat = [x / repeat for x in ts_cheat]
    #its = [x / repeat for x in its]
    qguts0 = [x / repeat for x in qguts0]
    #qguts1 = [x / repeat for x in qguts1]
    #qguts2 = [x / repeat for x in qguts2]
    qguts3 = [x / repeat for x in qguts3]
    qts_cheat = [x / repeat for x in qts_cheat]
    #qits = [x / repeat for x in qits]
    
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    #p_us = plt.plot(range(horizon), its, color='blue')
    #p_ts = plt.plot(range(horizon), guts2, color='black')
    #p_naive = plt.plot(range(horizon), guts1, '--', color='black')
    p_tsch = plt.plot(range(horizon), guts0, ':', color='black')
    p_ch = plt.plot(range(horizon), guts3, color='black')
    p_chc = plt.plot(range(horizon), ts_cheat, '-.', color='green')
    # p_random = plt.plot(range(n),[0.340195531946*x for x in range(n)], color='red')
    # p_random = plt.plot(range(horizon),[0.21329954363*x for x in range(horizon)], color='red')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.legend((p_tsch[0], #p_naive[0], p_ts[0], 
                p_ch[0], 
                p_chc[0]),
                     ('GUTS s0', 
                      #'GUTS s0c1', 'GUTS s1c0', 
                      'GUTS s1', 
                      'cheat TS'),
                     loc='upper left', fontsize=8)
    # leg.get_frame().set_alpha(0.5)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_g0 = plt.plot(range(horizon), qguts0, ':', color='black')
    #p_g1 = plt.plot(range(horizon), qguts1, '--', color='black')
    #p_g2 = plt.plot(range(horizon), qguts2, color='black')
    p_g3 = plt.plot(range(horizon), qguts3, color='black')
    #p_its = plt.plot(range(horizon), qits, '--', color='blue')
    #p_cheat = plt.plot(range(horizon), qts_cheat, color='green')
    #plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'),
    #           loc='upper left')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. queries')
    plt.show()

if __name__ == "__main__":

    experimentExmple5(200, 5, seed_=117)
    #experimentExample5significance(200, 5, seed_=117)
    #experimentRandomRandom(500, 5, seed_=45)
    #experimentVariantsRandom(1000, 5, seed_=45)
    #experimentVariantsRandom(1000, 5, seed_=145)
    #experimentVariantsRandom(1000, 5, arm_variance=0.05, seed_=777)
    #experimentVariantsRandom(1500, 5, arm_variance=0.05, no_objectives=2, cd_=10, seed_=108)
    #experimentVariantsRandom(1500, 5, arm_variance=0.05, no_objectives=4, cd_=10, seed_=411)
    #experimentVariantsRandom(1500, 5, arm_variance=0.05, no_objectives=6, cd_=10, seed_=117)
    sys.exit()
    
    # settings
    num_objectives = 2
    utility_std = 0.01 #was 0.01
    seed = 23097 #107 #123 #42
    num_iter = 200
    print_logs = True
    cool = 1
    initcool = 0
    sig=True
    sigth=0.01 #was 0.01
    
    arm_variance = 0.05

    add_virtual_comp = True
    temp_linear_prior = False
    gt=False
    
    #terms = [[(0,2)],[(0,1),(1,1)],[(1,2)]]
    #coeffs= [1.0, 3.0, 0.5]
    #utility_function = utilities.lambda_polynomial(terms, coeffs, seed=seed)
    
    #Random polynomial:
    #utility_function = utilities.random_polynomial_of_order_n(3,2,0.3,1.2)
    
    #Polynomial for the example problem
    terms = [[(0,1),(1,1)]]
    coeffs= [6.25]
    utility_function = utilities.lambda_polynomial(terms, coeffs, cd_=10, seed=seed)

    # initialise decision maker
    decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
    #print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))

    # initialise ground truth bandit
    #mabby = bandits.BernoulliBandit(2, 20, 5, seed)
    mabby = bandits.GaussianBandit(2, 20, varStd=arm_variance, predefined_seed=seed)
    #stats = bandits.GaussianBanditStats(mabby)
    mabby.redef_self_5arm_example()
    print('True utilities:', list(map(lambda x : decision_maker.true_utility(x), mabby.arms)))

    print(mabby.arms)
#    x,y = mabby.two_d_plot_lists()
#    plt.figure(num=None, figsize=(2.5, 2.5), dpi=80, facecolor='w', edgecolor='k')
#    plt.plot(x, y, 'ro')
#    plt.axis([-0.1,1.1,-0.1,1.1])
#    plt.xlabel(' $\mu^0_a$')
#    plt.ylabel(' $\mu^1_a$')
#    plt.show()
#    sys.exit()

    # run vanilla ITS:
    gt_vec_ = list(map(decision_maker.true_utility, mabby.arms))
    gt_vec = (gt_vec_ - min(gt_vec_)) / (max(gt_vec_) - min(gt_vec_))
    true_optimum = max(gt_vec)
    print(true_optimum)
    regret_vec = [true_optimum - x for x in gt_vec]
    linear_dm = LinearDecisionMaker(num_objectives, utility_std, defer_comp_to=decision_maker)
    regret_its, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, linear_dm, num_iter, prespecified_regret_vector=regret_vec)

    # run gp-ITS 
    add_virtual_comp = True 
    sig = True
    cool=0
    decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
    print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))

    regret, pull_counts, num_questions = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
    
    # run gp-ITS
    add_virtual_comp = False   
    cool=0
    decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
    print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))

    regret_, pull_counts_, num_questions_ = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)

    
    sys.exit()
    # run gp-ITS
    add_virtual_comp = True 
    sig = True
    cool = 0    
    decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
    print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))

    regret_2, pull_counts_2, num_questions_2 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)
    
    # run gp-ITS
    add_virtual_comp = True 
    sig = True
    cool = 3    
    decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
    print('True utility [0.5, 0.5]:', decision_maker.true_utility(np.array([0.5, 0.5])))

    regret_3, pull_counts_3, num_questions_3 = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)

    
    
    # run cheat TS
    gt=True     
    decision_maker = DecisionMaker(num_objectives, seed, utility_function, user_std=utility_std, temp_linear_prior=temp_linear_prior, add_virtual_comp=add_virtual_comp)
    regret_ch, pull_counts_ch, num_questions_ch = gp_utility_thompson_sampling(mabby, decision_maker, num_iter, print_logs=print_logs,
                                                                      cool=cool, initcool=initcool, sig_test=sig, sig_threshold=sigth,
                                                                      ground_truth=gt)

    # plot results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    p_ts = plt.plot(range(num_iter), np.cumsum(regret), '--', color='black')
    p_ts_ = plt.plot(range(num_iter), np.cumsum(regret_), color='black')
    p_ts_2 = plt.plot(range(num_iter), np.cumsum(regret_2), ':', color='black')
    p_ts_3 = plt.plot(range(num_iter), np.cumsum(regret_3), '-', color='grey')
    p_vanilla = plt.plot(range(num_iter), regret_its, '--', color='blue')
    p_ts = plt.plot(range(num_iter), np.cumsum(regret_ch), '-.', color='green')
    plt.xlabel('time')
    plt.ylabel('regret')

    plt.subplot(1, 2, 2)
    p_qs = plt.plot(range(num_iter), np.cumsum(num_questions), '--', color='black')
    p_qs_ = plt.plot(range(num_iter), np.cumsum(num_questions_),  color='black')
    p_qs_2 = plt.plot(range(num_iter), np.cumsum(num_questions_2), ':', color='black')
    p_qs_3 = plt.plot(range(num_iter), np.cumsum(num_questions_3), '-', color='gray')
    p_qs_ch = plt.plot(range(num_iter), qumap, '--', color='blue')
    plt.xlabel('time')
    plt.ylabel('no. questions')

    plt.tight_layout()
    plt.show()
    
    sys.exit()
    
        #test multivariate gaussian bandit
    mabby = bandits.GaussianBandit(2, 5, varStd=0.15)
    stats = bandits.GaussianBanditStats(mabby)
    print(mabby.arms[0])
    print(mabby.covarianceMatrices[0])
    pointsx, pointsy = [], []
    meanx, meany = [], []
    mu_0 = 0.5*np.ones(2)
    meanx.append(mu_0[0])
    meany.append(mu_0[1])
    sampx, sampy = [], []
    for i in range(500):
        smpl = stats.sample_mean(0,np.identity(2),mu_0,1,1)
        sampx.append(smpl[0])
        sampy.append(smpl[1])
        res = mabby.pull(0)
        stats.enter_data(0,res)
        pointsx.append(res[0])
        pointsy.append(res[1])
        #print(stats.scatter_matrix(0))
        
        #print(stats.sample_invwishart(0,2*np.identity(2),mu_0,1,1))
        mn = stats.mean_posterior(0,mu_0,1)
        print(mn)
        meanx.append(mn[0])
        meany.append(mn[1])
    pointsx2, pointsy2 = [], []
    for i in range(500):
        res = mabby.pull(1)
        stats.enter_data(1,res)
        pointsx2.append(res[0])
        pointsy2.append(res[1])
    pointsx3, pointsy3 = [], []
    for i in range(500):
        res = mabby.pull(2)
        stats.enter_data(2,res)
        pointsx3.append(res[0])
        pointsy3.append(res[1])
    pointsx4, pointsy4 = [], []
    for i in range(500):
        res = mabby.pull(3)
        stats.enter_data(3,res)
        pointsx4.append(res[0])
        pointsy4.append(res[1])
    pointsx5, pointsy5 = [], []
    for i in range(500):
        res = mabby.pull(4)
        stats.enter_data(4,res)
        pointsx5.append(res[0])
        pointsy5.append(res[1])
    plt.figure(figsize=(5, 5))
    plt.xlim([-.2,1.2])
    plt.ylim([-.2,1.2])
    plt.plot(pointsx, pointsy, 'ro', color='black')
    plt.plot(sampx[421:440], sampy[421:440], 'ro', color='red')
    #plt.plot(pointsx2, pointsy2, 'ro', color='green')
    #plt.plot(pointsx3, pointsy3, 'ro', color='red')
    #plt.plot(pointsx4, pointsy4, 'ro', color='blue')
    #plt.plot(pointsx5, pointsy5, 'ro', color='purple')
    #plt.plot(meanx[498:499], meany[498:499], 'ro', color='yellow')
    plt.xlabel('v0')
    plt.ylabel('v1')
    plt.show()
    
    print("arm 0: "+str(stats.current_estimate(0)))
    print("arm 1: "+str(stats.current_estimate(1)))
    print("arm 2: "+str(stats.current_estimate(2)))
    print("arm 3: "+str(stats.current_estimate(3)))
    print("arm 4: "+str(stats.current_estimate(4)))
    
    mat = np.repeat(np.transpose(np.repeat(0.5, 2)), 2, axis=1)
    print(mat)
    
    sys.exit()
