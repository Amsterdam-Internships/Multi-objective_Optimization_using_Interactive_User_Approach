#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:56:59 2017

@author: Diederik M. Roijers (Vrije Universiteit Brussel)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from sklearn import linear_model
import bayes_logistic as bl
import random

from bandits import BernoulliBandit as GroundTruthBandit
from bandits import BernoulliBanditStats as BanditStats
from bandits import sample_weight_vector


class LinearDecisionMaker:
    def __init__(self, no_obj, noise=0.0, predefined_weights=None, defer_comp_to=None):
        self.weights = np.zeros(no_obj)
        self.irrationality_sigma = noise
        self.previous_comparisons = []
        self.previous_outcomes = []
        self.defer_to_DM = defer_comp_to
        if predefined_weights is not None:
            self.weights = predefined_weights
        else:
            new_seed = math.floor(random.random() * 1000)
            random.seed(new_seed)
            sum_w = 0.0
            for j in range(no_obj):
                rnd = random.random()
                self.weights[j] = rnd
                sum_w = sum_w + rnd
            for j in range(no_obj):
                self.weights[j] = self.weights[j] / sum_w
                # print("DM LINWEIGHTS: ", self.weights)

    def exact_scalarise(self, vector):
        return np.inner(self.weights, vector)

    def noisy_scalarise(self, vector):
        if self.irrationality_sigma <= 0:
            return np.inner(self.weights, vector)
        noise = np.random.normal(0, self.irrationality_sigma)
        return np.inner(self.weights, vector) + noise

    def log_comparison(self, vector1, vector2, comp):
        diff_vector = vector1 - vector2
        res = 0.0
        if comp:
            res = 1.0
        self.previous_comparisons.append(diff_vector)
        self.previous_outcomes.append(res)

    def exact_compare(self, vector1, vector2, log=False):
        scalar1 = self.exact_scalarise(vector1)
        scalar2 = self.exact_scalarise(vector2)
        comp = (scalar1 >= scalar2)
        if log:
            self.log_comparison(vector1, vector2, comp)
        return comp

    def noisy_compare(self, vector1, vector2, log=True):
        if(self.defer_to_DM is None):
            scalar1 = self.noisy_scalarise(vector1)
            scalar2 = self.noisy_scalarise(vector2)
            comp = (scalar1 >= scalar2)
            if log:
                self.log_comparison(vector1, vector2, comp)
            return comp
        else:
           comp = self.defer_to_DM.noisy_compare(vector1, vector2, dont_update=True)
           if log:
                self.log_comparison(vector1, vector2, comp)
           return comp

    def reset_log(self):
        self.previous_comparisons = []
        self.previous_outcomes = []

    def current_ml_weights(self):
        if len(self.previous_outcomes) > 0:
            try:
                clf = linear_model.LogisticRegression(C=1e5)
                clf.fit(self.previous_comparisons, self.previous_outcomes)
                unnorm_w = clf.coef_[0]
            except:
                unnorm_w = np.array([random.random() for _ in range(len(self.weights))])
            sum_w = sum(unnorm_w)
            return unnorm_w / sum_w
        else:
            result = np.ones(len(self.weights))
            for i in range(len(self.weights)):
                result[i] = result[i] / float(len(self.weights))
            for i in range(len(self.weights)):
                if result[i] > 1.0:
                    new_result = np.zeros(len(self.weights))
                    new_result[i] = 1.0
                    return new_result
            return result

    def current_map(self):
        if len(self.previous_outcomes) > 0:
            # try :
            # clf = linear_model.LogisticRegression(C=1e5)
            # clf.fit(self.previous_comparisons, self.previous_outcomes)
            # unnorm_w = clf.coef_[0]
            w_prior = np.ones(len(self.weights)) / len(self.weights)
            H_prior_diag = np.ones(len(self.weights)) * (1.0 / 0.33) ** 2
            w_fit, H_fit = bl.fit_bayes_logistic(np.array(self.previous_outcomes),
                                                 np.array(self.previous_comparisons),
                                                 w_prior,
                                                 H_prior_diag)
            unnorm_w = w_fit
            # except:
            #     unnorm_w = np.array([random.random() for x in range(len(self.weights))])
            #     w_fit = unnorm_w
            #     H_fit = None
            sum_w = sum(unnorm_w)
            return unnorm_w / sum_w, w_fit, H_fit
        else:
            result = np.ones(len(self.weights))
            for i in range(len(self.weights)):
                result[i] = result[i] / float(len(self.weights))
            return result, result, None



def mo_ucb1_umap(bandit, decision_maker, n_max):
    stats = BanditStats(bandit)
    regret = 0
    questions = 0
    q_over_time = []
    true_weights = decision_maker.weights
    true_arm_number, true_optimum = bandit.gt_max(true_weights)
    gt_vec = bandit.gt_scalarisation_vector(true_weights)
    regret_vec = [true_optimum - x for x in gt_vec]
    regret_at_t = []
    weight_distances = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        # weights = decision_maker.current_ml_weights()
        weights, w_fit, H_fit = decision_maker.current_map()
        w_dist = math.sqrt(sum([x ** 2 for x in (true_weights - weights)]))
        weight_distances.append(w_dist)
        am, vm = stats.current_max(weights)
        au, vu = stats.current_ucb1_max(weights)
        if am != au:
            ru = bandit.pull(au)
            stats.enter_data(au, ru)
            em = stats.current_estimate(am)
            eu = stats.current_estimate(au)
            decision_maker.noisy_compare(em, eu)
            regret = regret + regret_vec[au]
            questions = questions + 1
        else:
            rm = bandit.pull(am)
            stats.enter_data(am, rm)
            em = stats.current_estimate(am)
            regret = regret + regret_vec[am]
        regret_at_t.append(regret)
        q_over_time.append(questions)
    return regret_at_t, weight_distances, list(stats.counts), q_over_time


def mo_ucbch_umap(bandit, decision_maker, n_max):
    stats = BanditStats(bandit)
    regret = 0
    questions = 0
    q_over_time = []
    true_weights = decision_maker.weights
    true_arm_number, true_optimum = bandit.gt_max(true_weights)
    gt_vec = bandit.gt_scalarisation_vector(true_weights)
    regret_vec = [true_optimum - x for x in gt_vec]
    regret_at_t = []
    weight_distances = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        # weights = decision_maker.current_ml_weights()
        weights, w_fit, H_fit = decision_maker.current_map()
        w_dist = math.sqrt(sum([x ** 2 for x in (true_weights - weights)]))
        weight_distances.append(w_dist)
        am, vm = stats.current_max(weights)
        au, vu = stats.current_ucbch_max(weights)
        if am != au:
            ru = bandit.pull(au)
            stats.enter_data(au, ru)
            em = stats.current_estimate(am)
            eu = stats.current_estimate(au)
            decision_maker.noisy_compare(em, eu)
            regret = regret + regret_vec[au]
            questions = questions + 1
        else:
            rm = bandit.pull(am)
            stats.enter_data(am, rm)
            em = stats.current_estimate(am)
            regret = regret + regret_vec[am]
        regret_at_t.append(regret)
        q_over_time.append(questions)
    return regret_at_t, weight_distances, list(stats.counts), q_over_time


def mo_ucb1_cheat_gt_weights(bandit, decision_maker, n_max):
    stats = BanditStats(bandit)
    regret = 0
    true_weights = decision_maker.weights
    true_arm_number, true_optimum = bandit.gt_max(true_weights)
    gt_vec = bandit.gt_scalarisation_vector(true_weights)
    regret_vec = [true_optimum - x for x in gt_vec]
    regret_at_t = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        weights = true_weights
        au, vu = stats.current_ucb1_max(weights)
        ru = bandit.pull(au)
        stats.enter_data(au, ru)
        #eu = stats.current_estimate(au)
        regret = regret + regret_vec[au]
        regret_at_t.append(regret)
    return regret_at_t, list(stats.counts)


def mo_ucbch_cheat_gt_weights(bandit, decision_maker, n_max):
    stats = BanditStats(bandit)
    regret = 0
    true_weights = decision_maker.weights
    true_arm_number, true_optimum = bandit.gt_max(true_weights)
    gt_vec = bandit.gt_scalarisation_vector(true_weights)
    regret_vec = [true_optimum - x for x in gt_vec]
    regret_at_t = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        weights = true_weights
        au, vu = stats.current_ucbch_max(weights)
        ru = bandit.pull(au)
        stats.enter_data(au, ru)
        #eu = stats.current_estimate(au)
        regret = regret + regret_vec[au]
        regret_at_t.append(regret)
    return regret_at_t, list(stats.counts)


def mo_ts_umap(bandit, decision_maker, n_max):
    """NB: not used in paper"""
    stats = BanditStats(bandit)
    regret = 0
    true_weights = decision_maker.weights
    true_arm_number, true_optimum = bandit.gt_max(true_weights)
    gt_vec = bandit.gt_scalarisation_vector(true_weights)
    regret_vec = [true_optimum - x for x in gt_vec]
    regret_at_t = []
    weight_distances = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        weights = decision_maker.current_ml_weights()
        w_dist = math.sqrt(sum([x ** 2 for x in (true_weights - weights)]))
        weight_distances.append(w_dist)
        am, vm = stats.current_max(weights)
        au, vu = stats.ts_sample_max(weights)
        if am != au:
            ru = bandit.pull(au)
            stats.enter_data(au, ru)
            em = stats.current_estimate(am)
            eu = stats.current_estimate(au)
            decision_maker.noisy_compare(em, eu)
            regret = regret + regret_vec[au]
        else:
            rm = bandit.pull(am)
            stats.enter_data(am, rm)
            em = stats.current_estimate(am)
            regret = regret + regret_vec[am]
        regret_at_t.append(regret)
    return regret_at_t, weight_distances, list(stats.counts)


def mo_thompson_cheat_gt_weights(bandit, decision_maker, n_max):
    stats = BanditStats(bandit)
    regret = 0
    true_weights = decision_maker.weights
    true_arm_number, true_optimum = bandit.gt_max(true_weights)
    gt_vec = bandit.gt_scalarisation_vector(true_weights)
    regret_vec = [true_optimum - x for x in gt_vec]
    regret_at_t = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        weights = true_weights
        au, vu = stats.ts_sample_max(weights)
        ru = bandit.pull(au)
        stats.enter_data(au, ru)
        #eu = stats.current_estimate(au)
        regret = regret + regret_vec[au]
        regret_at_t.append(regret)
    return regret_at_t, list(stats.counts)


def interactive_thompson_sampling(bandit, decision_maker, n_max, prespecified_regret_vector=None):
    stats = BanditStats(bandit)
    regret = 0
    questions = 0
    q_over_time = []    
    regret_vec = []
    if(prespecified_regret_vector is None):
        true_weights = decision_maker.weights
        true_arm_number, true_optimum = bandit.gt_max(true_weights)
        gt_vec = bandit.gt_scalarisation_vector(true_weights)
        regret_vec = [true_optimum - x for x in gt_vec]
    else:
       regret_vec =  prespecified_regret_vector    
    regret_at_t = []
    weight_distances = []
    # print("Max: ", sum(regret_vec)/len(regret_vec))
    for i in range(n_max):
        weights, w_fit, H_fit = decision_maker.current_map()
        w_dist = 0
        if(prespecified_regret_vector is None):
            w_dist = math.sqrt(sum([x ** 2 for x in (true_weights - weights)]))
        weight_distances.append(w_dist)
        w1 = sample_weight_vector(w_fit, H_fit)
        w2 = sample_weight_vector(w_fit, H_fit)
        am, vm = stats.ts_sample_max(w1)
        au, vu = stats.ts_sample_max(w2)
        if am != au:
            ru = bandit.pull(au)
            stats.enter_data(au, ru)
            em = stats.current_estimate(am)
            eu = stats.current_estimate(au)
            decision_maker.noisy_compare(em, eu)
            regret = regret + regret_vec[au]
            questions = questions + 1
        else:
            rm = bandit.pull(am)
            stats.enter_data(am, rm)
            em = stats.current_estimate(am)
            regret = regret + regret_vec[am]
        regret_at_t.append(regret)
        q_over_time.append(questions)
    return regret_at_t, weight_distances, list(stats.counts), q_over_time


def experimentDoubleCircle(horizon, repeat, noise, weight=None):
    figx = 3.0
    figy = 2.1
    dpix = 120
    no_objectives = 2
    no_arms = 30
    normaliser = 5
    the_seed = 181
    the_dm = LinearDecisionMaker(no_objectives, noise, np.array([0.2, 0.8]))
    mabby = GroundTruthBandit(no_objectives, no_arms, normaliser, the_seed)
    mabby.redef_self_circular(10, 2, 0.7)
    ucb1 = [0] * horizon
    ucbch = [0] * horizon
    dts = [0] * horizon
    ucb1c = [0] * horizon
    ucbcc = [0] * horizon
    dtsc = [0] * horizon
    qucb1 = [0] * horizon
    qucbch = [0] * horizon
    qdts = [0] * horizon
    wucb1 = [0] * horizon
    wucbch = [0] * horizon
    wdts = [0] * horizon
    f = open('datalog.py', 'a')
    for i in range(repeat):
        regret, w_dists, cnts, qumap = mo_ucb1_umap(mabby, the_dm, horizon)
        ucb1 = [ucb1[cnt] + regret[cnt] for cnt in range(len(ucb1))]
        qucb1 = [qucb1[cnt] + qumap[cnt] for cnt in range(len(qucb1))]
        wucb1 = [wucb1[cnt] + w_dists[cnt] for cnt in range(len(wucb1))]
        f.write("u1r_" + str(i) + " = " + str(ucb1) + "\n")
        f.write("u1q_" + str(i) + " = " + str(qucb1) + "\n")
        f.write("u1w_" + str(i) + " = " + str(wucb1) + "\n")
        the_dm.reset_log()
        regret, w_dists, cnts, qumap = mo_ucbch_umap(mabby, the_dm, horizon)
        ucbch = [ucbch[cnt] + regret[cnt] for cnt in range(len(ucbch))]
        qucbch = [qucbch[cnt] + qumap[cnt] for cnt in range(len(qucbch))]
        wucbch = [wucbch[cnt] + w_dists[cnt] for cnt in range(len(wucbch))]
        f.write("c1r_" + str(i) + " = " + str(ucbch) + "\n")
        f.write("c1q_" + str(i) + " = " + str(qucbch) + "\n")
        f.write("c1w_" + str(i) + " = " + str(wucbch) + "\n")
        the_dm.reset_log()
        regret, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, the_dm, horizon)
        dts = [dts[cnt] + regret[cnt] for cnt in range(len(dts))]
        qdts = [qdts[cnt] + qumap[cnt] for cnt in range(len(qdts))]
        wdts = [wdts[cnt] + w_dists[cnt] for cnt in range(len(wdts))]
        f.write("d1r_" + str(i) + " = " + str(dts) + "\n")
        f.write("d1q_" + str(i) + " = " + str(qdts) + "\n")
        f.write("d1w_" + str(i) + " = " + str(wdts) + "\n")
        the_dm.reset_log()
        regret, cnts = mo_ucb1_cheat_gt_weights(mabby, the_dm, horizon)
        ucb1c = [ucb1c[cnt] + regret[cnt] for cnt in range(len(ucb1c))]
        f.write("uc1r_" + str(i) + " = " + str(ucb1c) + "\n")
        the_dm.reset_log()
        regret, cnts = mo_ucbch_cheat_gt_weights(mabby, the_dm, horizon)
        ucbcc = [ucbcc[cnt] + regret[cnt] for cnt in range(len(ucbcc))]
        f.write("cc1r_" + str(i) + " = " + str(ucbcc) + "\n")
        the_dm.reset_log()
        regret, cnts = mo_thompson_cheat_gt_weights(mabby, the_dm, horizon)
        dtsc = [dtsc[cnt] + regret[cnt] for cnt in range(len(dtsc))]
        f.write("dc1r_" + str(i) + " = " + str(dtsc) + "\n")
    f.close()
    ucb1 = [x / repeat for x in ucb1]
    ucbch = [x / repeat for x in ucbch]
    dts = [x / repeat for x in dts]
    ucb1c = [x / repeat for x in ucb1c]
    ucbcc = [x / repeat for x in ucbcc]
    dtsc = [x / repeat for x in dtsc]
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), ucb1, color='blue')
    p_ts = plt.plot(range(horizon), dts, color='black')
    p_naive = plt.plot(range(horizon), ucb1c, '--', color='blue')
    p_tsch = plt.plot(range(horizon), dtsc, '--', color='black')
    p_ch = plt.plot(range(horizon), ucbch, color='green')
    p_chc = plt.plot(range(horizon), ucbcc, '--', color='green')
    # p_random = plt.plot(range(n),[0.340195531946*x for x in range(n)], color='red')
    # p_random = plt.plot(range(horizon),[0.21329954363*x for x in range(horizon)], color='red')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.legend((p_us[0], p_ch[0], p_ts[0], p_naive[0], p_chc[0], p_tsch[0]),
                     ('umap-UCB1', 'umap-UCBch', 'DTS', 'cheat UCB1', 'cheat UCBch', 'cheat TS'),
                     loc='upper left', fontsize=8)
    # leg.get_frame().set_alpha(0.5)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    qucb1 = [x / repeat for x in qucb1]
    qucbch = [x / repeat for x in qucbch]
    qdts = [x / repeat for x in qdts]
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), qucb1, color='blue')
    p_ts = plt.plot(range(horizon), qdts, color='black')
    p_ch = plt.plot(range(horizon), qucbch, color='green')
    # plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'),
    #           loc='upper left')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. comparisons')
    plt.show()

    wucb1 = [x / repeat for x in wucb1]
    wucbch = [x / repeat for x in wucbch]
    wdts = [x / repeat for x in wdts]
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), wucb1, color="blue")
    p_ts = plt.plot(range(horizon), wdts, color="black")
    p_ch = plt.plot(range(horizon), wucbch, color="green")
    # plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'))
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('|w*-w|')
    plt.show()


def experimentRandom(noobj, horizon, repeat, noise):
    figx = 3.0
    figy = 2.1
    dpix = 120
    no_objectives = noobj
    no_arms = 30
    normaliser = 5
    the_seed = 181
    the_dm = LinearDecisionMaker(no_objectives, noise, np.array([0.2, 0.8]))
    mabby = GroundTruthBandit(no_objectives, no_arms, normaliser, the_seed)
    ucb1 = [0] * horizon
    ucbch = [0] * horizon
    dts = [0] * horizon
    ucb1c = [0] * horizon
    ucbcc = [0] * horizon
    dtsc = [0] * horizon
    qucb1 = [0] * horizon
    qucbch = [0] * horizon
    qdts = [0] * horizon
    wucb1 = [0] * horizon
    wucbch = [0] * horizon
    wdts = [0] * horizon
    f = open('randomlog.py', 'a')
    for i in range(repeat):
        wvec = np.zeros(no_objectives)
        wsum = 0
        for z in range(no_objectives):
            wvec[z] = random.random()
            wsum = wsum + wvec[z]
        for z in range(no_objectives):
            wvec[z] = wvec[z] / wsum
        the_dm = LinearDecisionMaker(no_objectives, noise, wvec)
        print(the_dm.weights)
        mabby = GroundTruthBandit(no_objectives, no_arms, normaliser)
        print(mabby.arms)
        regret, w_dists, cnts, qumap = mo_ucb1_umap(mabby, the_dm, horizon)
        ucb1 = [ucb1[cnt] + regret[cnt] for cnt in range(len(ucb1))]
        qucb1 = [qucb1[cnt] + qumap[cnt] for cnt in range(len(qucb1))]
        wucb1 = [wucb1[cnt] + w_dists[cnt] for cnt in range(len(wucb1))]
        f.write("u1r_" + str(i) + " = " + str(ucb1) + "\n")
        f.write("u1q_" + str(i) + " = " + str(qucb1) + "\n")
        f.write("u1w_" + str(i) + " = " + str(wucb1) + "\n")
        the_dm.reset_log()
        regret, w_dists, cnts, qumap = mo_ucbch_umap(mabby, the_dm, horizon)
        ucbch = [ucbch[cnt] + regret[cnt] for cnt in range(len(ucbch))]
        qucbch = [qucbch[cnt] + qumap[cnt] for cnt in range(len(qucbch))]
        wucbch = [wucbch[cnt] + w_dists[cnt] for cnt in range(len(wucbch))]
        f.write("c1r_" + str(i) + " = " + str(ucbch) + "\n")
        f.write("c1q_" + str(i) + " = " + str(qucbch) + "\n")
        f.write("c1w_" + str(i) + " = " + str(wucbch) + "\n")
        the_dm.reset_log()
        regret, w_dists, cnts, qumap = interactive_thompson_sampling(mabby, the_dm, horizon)
        dts = [dts[cnt] + regret[cnt] for cnt in range(len(dts))]
        qdts = [qdts[cnt] + qumap[cnt] for cnt in range(len(qdts))]
        wdts = [wdts[cnt] + w_dists[cnt] for cnt in range(len(wdts))]
        f.write("d1r_" + str(i) + " = " + str(dts) + "\n")
        f.write("d1q_" + str(i) + " = " + str(qdts) + "\n")
        f.write("d1w_" + str(i) + " = " + str(wdts) + "\n")
        the_dm.reset_log()
        regret, cnts = mo_ucb1_cheat_gt_weights(mabby, the_dm, horizon)
        ucb1c = [ucb1c[cnt] + regret[cnt] for cnt in range(len(ucb1c))]
        f.write("uc1r_" + str(i) + " = " + str(ucb1c) + "\n")
        the_dm.reset_log()
        regret, cnts = mo_ucbch_cheat_gt_weights(mabby, the_dm, horizon)
        ucbcc = [ucbcc[cnt] + regret[cnt] for cnt in range(len(ucbcc))]
        f.write("cc1r_" + str(i) + " = " + str(ucbcc) + "\n")
        the_dm.reset_log()
        regret, cnts = mo_thompson_cheat_gt_weights(mabby, the_dm, horizon)
        dtsc = [dtsc[cnt] + regret[cnt] for cnt in range(len(dtsc))]
        f.write("dc1r_" + str(i) + " = " + str(dtsc) + "\n")
    f.close()
    ucb1 = [x / repeat for x in ucb1]
    ucbch = [x / repeat for x in ucbch]
    dts = [x / repeat for x in dts]
    ucb1c = [x / repeat for x in ucb1c]
    ucbcc = [x / repeat for x in ucbcc]
    dtsc = [x / repeat for x in dtsc]
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), ucb1, color='blue')
    p_ts = plt.plot(range(horizon), dts, color='black')
    p_naive = plt.plot(range(horizon), ucb1c, '--', color='blue')
    p_tsch = plt.plot(range(horizon), dtsc, '--', color='black')
    p_ch = plt.plot(range(horizon), ucbch, color='green')
    p_chc = plt.plot(range(horizon), ucbcc, '--', color='green')
    # p_random = plt.plot(range(n),[0.340195531946*x for x in range(n)], color='red')
    # p_random = plt.plot(range(horizon),[0.21329954363*x for x in range(horizon)], color='red')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    leg = plt.legend((p_us[0], p_ch[0], p_ts[0], p_naive[0], p_chc[0], p_tsch[0]),
                     ('umap-UCB1', 'umap-UCBch', 'DTS', 'cheat UCB1', 'cheat UCBch', 'cheat TS'),
                     loc='upper left', fontsize=9)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    qucb1 = [x / repeat for x in qucb1]
    qucbch = [x / repeat for x in qucbch]
    qdts = [x / repeat for x in qdts]
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), qucb1, color='blue')
    p_ts = plt.plot(range(horizon), qdts, color='black')
    p_ch = plt.plot(range(horizon), qucbch, color='green')
    # plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'),
    #           loc='upper left')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. comparisons')
    plt.show()

    wucb1 = [x / repeat for x in wucb1]
    wucbch = [x / repeat for x in wucbch]
    wdts = [x / repeat for x in wdts]
    plt.figure(num=None, figsize=(figx, figy), dpi=dpix, facecolor='w', edgecolor='k')
    p_us = plt.plot(range(horizon), wucb1, color="blue")
    p_ts = plt.plot(range(horizon), wdts, color="black")
    p_ch = plt.plot(range(horizon), wucbch, color="green")
    # plt.legend( (p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch','DTS'))
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('|w*-w|')
    plt.show()


if __name__ == "__main__":
    print("Welcome to this experiment in Multi-Objective Multi-Armed Bandits")
    # experimentDoubleCircle(2000, 10, 0.001)
    experimentRandom(2, 2000, 3, 0.001)
    sys.exit()
    x = [0] * 10
    print(x)
    ### BASIC EXPERIMENTAL SETTINGS ###
    no_objectives = 2
    no_arms = 30
    normaliser = 5
    the_seed = 181
    the_dm = LinearDecisionMaker(no_objectives, 0.0, np.array([0.34, 0.66]))
    ### CREATE EXPERIMENT ###
    mabby = GroundTruthBandit(no_objectives, no_arms, normaliser, the_seed)
    print(mabby.arms)
    i = 5
    print("arm ", i, ":", mabby.arms[i])
    x, y = mabby.two_d_plot_lists()
    estimate_total = mabby.pull(i)
    """
    for cnt in range(500):
        vec = mabby.pull(i)
        estimate_total = estimate_total + vec
        estimate = estimate_total/(cnt+2.0)
        if( cnt%100 == 0):
            plt.plot(x, y, 'ro', 
                     [mabby.arms[i][0]],[mabby.arms[i][1]],'go',
                     [estimate[0]], [estimate[1]], 'bo')
            plt.axis([-0.1,1.1,-0.1,1.1])
            plt.ylabel('V[0]')
            plt.ylabel('V[1]')
            plt.show()
            print("Estimate:", estimate, " difference:", estimate-mabby.arms[i])
    """
    mabby.redef_self_circular(10, 2, 0.7)
    # print(mabby.arms)
    # x,y = mabby.two_d_plot_lists()
    # plt.figure(num=None, figsize=(2.5, 2.5), dpi=80, facecolor='w', edgecolor='k')
    # plt.plot(x, y, 'ro')
    # plt.axis([-0.1,1.1,-0.1,1.1])
    # plt.xlabel(' $\mu^0_a$')
    # plt.ylabel(' $\mu^1_a$')
    # plt.show()
    # sys.exit()

    true_weights = the_dm.weights
    true_arm_number, true_optimum = mabby.gt_max(true_weights)
    gt_vec = mabby.gt_scalarisation_vector(true_weights)
    # regret_vec = [true_optimum - x for x in gt_vec]
    x = range(20)
    width = 0.5
    plt.bar(x, gt_vec, width, color="red")
    plt.show()

    n = 1000
    regret, w_dists, cnts, qumap = mo_ucb1_umap(mabby, the_dm, n)
    # print(len(the_dm.previous_outcomes))
    # print(sum(the_dm.previous_outcomes))
    # print(sum(the_dm.previous_outcomes)/(1.0*len(the_dm.previous_outcomes)))
    # print(the_dm.current_ml_weights())
    # normmap, wmap, hmap = the_dm.current_map()
    # print(normmap)
    # print(wmap)
    # print(hmap)
    # sys.exit()
    the_dm.reset_log()
    regret_ch, w_dists_ch, cnts_ch, qdch = mo_ucbch_umap(mabby, the_dm, n)
    the_dm.reset_log()
    regret_ts, w_dists_ts, cnts_ts, qdts = interactive_thompson_sampling(mabby, the_dm, n)
    x = range(20)
    width = 0.5
    # plt.bar(x, cnts, width, color="blue")
    # plt.bar(x, cnts_ts, width, color="red")
    # plt.show()
    the_dm.reset_log()
    regret_cheat, cnts_cheat = mo_ucb1_cheat_gt_weights(mabby, the_dm, n)
    the_dm.reset_log()
    regret_tscheat, cnts_tscheat = mo_thompson_cheat_gt_weights(mabby, the_dm, n)
    x = range(20)
    width = 0.5
    # plt.bar(x, cnts_cheat, width, color="gray")
    # plt.bar(x, cnts, width, color="blue")
    # plt.bar(x, cnts_ts, width, color="red")
    # plt.show()
    p_us = plt.plot(range(n), regret, color='blue')
    p_ts = plt.plot(range(n), regret_ts, color='black')
    p_naive = plt.plot(range(n), regret_cheat, color='gray')
    p_tsch = plt.plot(range(n), regret_tscheat, color='green')
    p_ch = plt.plot(range(n), regret_ch, color='purple')
    p_random = plt.plot(range(n), [0.340195531946 * x for x in range(n)], color='red')
    p_random = plt.plot(range(n), [0.21329954363 * x for x in range(n)], color='red')
    plt.legend((p_us[0], p_ch[0], p_ts[0], p_naive[0], p_tsch[0], p_random[0]),
               ('umap-UCB1', 'umap-UCBch', 'DTS', 'cheat UCB', 'cheat TS', 'random'))
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.show()

    p_us = plt.plot(range(n), qumap, color='blue')
    p_ts = plt.plot(range(n), qdts, color='black')
    p_ch = plt.plot(range(n), qdch, color='purple')
    plt.legend((p_us[0], p_ch[0], p_ts[0]), ('umap-UCB1', 'umap-UCBch', 'DTS'))
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('no. comparisons')
    plt.show()

    plt.plot(range(n), w_dists, color="blue")
    plt.plot(range(n), w_dists_ts, color="black")
    plt.plot(range(n), w_dists_ch, color="purple")
    # plt.axis([-0.1,1.1,-0.1,1.1])
    plt.xlabel('time')
    plt.ylabel('|w*-w|')
    plt.show()
