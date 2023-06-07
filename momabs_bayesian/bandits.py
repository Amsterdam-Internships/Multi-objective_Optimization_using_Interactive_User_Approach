#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:42:45 2017

@author: Diederik M. Roijers (Vrije Universiteit Brussel)
"""

import numpy as np
from scipy.stats import beta
import scipy
import random
import math
import functools
import operator
import spm1d.stats.t
import spm1d.stats.hotellings

class BernoulliBandit:
    """
    This class contains the ground thruth bandit, i.e., the true environment.
    The main function of this class is pull, parameterised by an arm number
    """

    def __init__(self, no_obj, no_arms, n_=5, predefined_seed=0):
        self.number_of_objectives = no_obj
        self.arms = []
        n = n_
        maxima = np.zeros(self.number_of_objectives)
        minima = np.ones(self.number_of_objectives)
        if predefined_seed != 0:
            random.seed(predefined_seed)
            np.random.seed(predefined_seed)
            print(" SEED: ", predefined_seed)
        else:
            new_seed = math.floor(random.random() * 1000)
            random.seed(new_seed)
            np.random.seed(new_seed)
            print("SEED: ", new_seed)
        for i in range(no_arms):
            # mu_vector = np.zeros(self.number_of_objectives)
            mu_vector = sample_weight_vector(np.ones(self.number_of_objectives),
                                             0.5 * np.ones(self.number_of_objectives))
            for j in range(self.number_of_objectives):
                sum_n = 0.0
                for k in range(n):
                    sum_n = sum_n + random.random()
                mu_vector[j] = sum_n / float(n)
                if mu_vector[j] > maxima[j]:
                    maxima[j] = mu_vector[j]
                if mu_vector[j] < minima[j]:
                    minima[j] = mu_vector[j]
            self.arms.append(mu_vector)
        for i in range(no_arms):
            for j in range(self.number_of_objectives):
                self.arms[i][j] = (self.arms[i][j] - minima[j]) / (maxima[j] - minima[j])

    def redef_self_circular(self, no_ticks, no_obj, sub_circle_radius=0.0):
        self.number_of_objectives = no_obj
        self.arms = []
        circle = 0.5 * math.pi
        tick = circle / (no_ticks - 1.0)
        lst = [[]]
        for i in range(no_obj - 1):
            res = []
            for j in range(len(lst)):
                item = lst[j]
                for k in range(no_ticks):
                    new_item = list(item)
                    new_item.append(k * tick)
                    res.append(new_item)
            lst = list(res)
        for item in lst:
            coord = []
            base = 1
            for phi in item:
                coord.append(math.cos(phi))
                base = base * math.sin(phi)
            coord.append(base)
            self.arms.append(coord)
        if sub_circle_radius > 0.0:
            for item in lst:
                coord = []
                base = 1
                for phi in item:
                    coord.append(sub_circle_radius * math.cos(phi))
                    base = base * math.sin(phi)
                coord.append(sub_circle_radius * base)
                self.arms.append(coord)

    def pull(self, arm_number, dist='bernoulli'):
        result = np.zeros(self.number_of_objectives)
        if dist == 'bernoulli':
            the_arm = self.arms[arm_number]
            for i in range(self.number_of_objectives):
                r_number = random.random()
                if r_number < the_arm[i]:
                    result[i] = 1.0
                else:
                    result[i] = 0.0
        return result

    def gt_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(len(self.arms)):
            scal_val = np.inner(weight_vector, self.arms[i])
            if scal_val >= c_max:
                arm = i
                c_max = scal_val
        return arm, c_max

    def gt_scalarisation_vector(self, weight_vector):
        vec = []
        for i in range(len(self.arms)):
            scal_val = np.inner(weight_vector, self.arms[i])
            vec.append(scal_val)
        return vec

    def two_d_plot_lists(self):
        x = []
        y = []
        for i in range(len(self.arms)):
            x.append(self.arms[i][0])
            y.append(self.arms[i][1])
        return x, y
    
    
class GaussianBandit:
    """
    This class contains multivariate Gaussian ground thruth bandit, i.e., 
    the true environment, with correlated normally distributed rewards.
    The main function of this class is pull, parameterised by an arm number.
    """

    def __init__(self, no_obj, no_arms, varStd=0.001, n_=5, predefined_seed=0):
        self.number_of_objectives = no_obj
        self.arms = []
        self.covarianceMatrices = []
        n = n_
        maxima = np.zeros(self.number_of_objectives)
        minima = np.ones(self.number_of_objectives)
        if predefined_seed != 0:
            random.seed(predefined_seed)
            np.random.seed(predefined_seed)
            print(" SEED: ", predefined_seed)
        else:
            new_seed = math.floor(random.random() * 1000)
            random.seed(new_seed)
            np.random.seed(new_seed)
            print("SEED: ", new_seed)
        for i in range(no_arms):
            # mu_vector = np.zeros(self.number_of_objectives)
            mu_vector = sample_weight_vector(np.ones(self.number_of_objectives),
                                             0.5 * np.ones(self.number_of_objectives))
            for j in range(self.number_of_objectives):
                sum_n = 0.0
                for k in range(n):
                    sum_n = sum_n + random.random()
                mu_vector[j] = sum_n / float(n)
                if mu_vector[j] > maxima[j]:
                    maxima[j] = mu_vector[j]
                if mu_vector[j] < minima[j]:
                    minima[j] = mu_vector[j]
            self.arms.append(mu_vector)
        for i in range(no_arms):
            covMatrix = np.zeros( (no_obj,no_obj) )
            for k in range(self.number_of_objectives):
                    covMatrix[k,k] = random.random()*varStd
            for j in range(self.number_of_objectives):
                self.arms[i][j] = (self.arms[i][j] - minima[j]) / (maxima[j] - minima[j])
                for k in range(j):
                    corr = 2*(random.random()-0.5) #correlation
                    #covariance j and k
                    css= corr*math.sqrt(covMatrix[k,k])*math.sqrt(covMatrix[j,j])
                    covMatrix[k,j] = css
                    covMatrix[j,k] = css
            self.covarianceMatrices.append(covMatrix)
                    

    def redef_self_circular(self, no_ticks, no_obj, varStd = 0.01, sub_circle_radius=0.0):
        self.number_of_objectives = no_obj
        self.arms = []
        self.covarianceMatrix = []
        circle = 0.5 * math.pi
        tick = circle / (no_ticks - 1.0)
        lst = [[]]
        for i in range(no_obj - 1):
            res = []
            for j in range(len(lst)):
                item = lst[j]
                for k in range(no_ticks):
                    new_item = list(item)
                    new_item.append(k * tick)
                    res.append(new_item)
            lst = list(res)
        for item in lst:
            coord = []
            base = 1
            for phi in item:
                coord.append(math.cos(phi))
                base = base * math.sin(phi)
            coord.append(base)
            self.arms.append(coord)
        if sub_circle_radius > 0.0:
            for item in lst:
                coord = []
                base = 1
                for phi in item:
                    coord.append(sub_circle_radius * math.cos(phi))
                    base = base * math.sin(phi)
                coord.append(sub_circle_radius * base)
                self.arms.append(coord)
        for i in range(len(self.arms)):
            covMatrix = np.zeros( (no_obj,no_obj) )
            for k in range(self.number_of_objectives):
                    covMatrix[k,k] = varStd
            self.covarianceMatrices.append(covMatrix)

    def pull(self, arm_number, dist='bernoulli'):
        result = np.random.multivariate_normal(self.arms[arm_number],
                                               self.covarianceMatrices[arm_number])
        return result
    
    def redef_self_5arm_example(self, varStd = 0.005):
        self.number_of_objectives = 2
        self.arms = []
        self.covarianceMatrix = []
        self.arms.append(np.array([0.0, 0.8]))
        self.arms.append(np.array([0.1, 0.9]))
        #self.arms.append(np.array([0.2, 0.6]))
        self.arms.append(np.array([0.4, 0.4]))
        #self.arms.append(np.array([0.6, 0.2]))
        self.arms.append(np.array([0.9, 0.1]))
        self.arms.append(np.array([0.8, 0.0]))
        for i in range(len(self.arms)):
            covMatrix = np.zeros( (2,2) )
            for k in range(self.number_of_objectives):
                    covMatrix[k,k] = varStd
            self.covarianceMatrices.append(covMatrix)
        

    def gt_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(len(self.arms)):
            scal_val = np.inner(weight_vector, self.arms[i])
            if scal_val >= c_max:
                arm = i
                c_max = scal_val
        return arm, c_max

    def gt_scalarisation_vector(self, weight_vector):
        vec = []
        for i in range(len(self.arms)):
            scal_val = np.inner(weight_vector, self.arms[i])
            vec.append(scal_val)
        return vec

    def two_d_plot_lists(self):
        x = []
        y = []
        for i in range(len(self.arms)):
            x.append(self.arms[i][0])
            y.append(self.arms[i][1])
        return x, y


class BernoulliBanditStats:
    def __init__(self, gt_bandit):
        self.total_count = 0.0
        self.sums = []
        self.counts = []
        self.no_arms = len(gt_bandit.arms)
        self.number_of_objectives = gt_bandit.number_of_objectives
        for i in range(self.no_arms):
            init_estimate = gt_bandit.pull(i)
            self.sums.append(init_estimate)
            self.counts.append(1.0)
            self.total_count = self.total_count + 1.0

    def enter_data(self, arm, sample):
        self.total_count = self.total_count + 1
        self.counts[arm] = self.counts[arm] + 1
        self.sums[arm] = self.sums[arm] + sample

    def mo_ucb1_value(self, arm, weight_vector):
        bonus = math.sqrt((2.0 * math.log(self.total_count)) / self.counts[arm])
        arm_val = self.sums[arm] / self.counts[arm]
        scal_val = np.inner(weight_vector, arm_val)
        return scal_val + bonus

    def mo_ucbch_value(self, arm, weight_vector):
        arm_val = self.sums[arm] / self.counts[arm]
        scal_val = np.inner(weight_vector, arm_val)
        bonusa = (scal_val * math.log(self.total_count)) / self.counts[arm]
        if bonusa <= 0.00000000001:
            bonusb = 0
        else:
            bonusb = math.sqrt(bonusa)
        bonusc = math.log(self.total_count) / self.counts[arm]
        bonus = bonusb + bonusc
        return scal_val + bonus

    def current_estimate(self, arm):
        return self.sums[arm] / self.counts[arm]

    def current_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(self.no_arms):
            arm_val = self.sums[i] / self.counts[i]
            scal_val = np.inner(weight_vector, arm_val)
            if scal_val >= c_max:
                arm = i
                c_max = scal_val
        return arm, c_max

    def current_ucb1_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(self.no_arms):
            ucb_val = self.mo_ucb1_value(i, weight_vector)
            if ucb_val >= c_max:
                arm = i
                c_max = ucb_val
        return arm, c_max

    def current_ucbch_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(self.no_arms):
            ucb_val = self.mo_ucbch_value(i, weight_vector)
            if ucb_val >= c_max:
                arm = i
                c_max = ucb_val
        return arm, c_max

    def mo_ts_sample_value(self, arm, weight_vector):
        sampled_means = []  # we assume that the mean for each objective is independent
        # i.e., the covariance matrix only has positive values on the diagonal
        successes = self.sums[arm]
        totals = self.counts[arm]
        for i in range(len(weight_vector)):
            a = successes[i] + 1
            a = max(a,1)
            b = totals - successes[i] + 1
            b = max(b,1)
            #print(a, b)
            sample = beta.rvs(a, b, size=1)  # draw from beta distribution
            sampled_means.append(sample[0])
        arm_val = np.array(sampled_means)
        scal_val = np.inner(weight_vector, arm_val)
        return scal_val

    def ts_sample_max(self, weight_vector):
        mean_max = 0
        arm = 0
        for i in range(self.no_arms):
            sampled_val = self.mo_ts_sample_value(i, weight_vector)
            if sampled_val >= mean_max:
                arm = i
                mean_max = sampled_val
        return arm, mean_max

    def ts_sample_arm(self, arm):
        sampled_means = []  # we assume that the mean for each objective is independent
        # i.e., the covariance matrix only has positive values on the diagonal
        successes = self.sums[arm]
        totals = self.counts[arm]
        for i in range(self.number_of_objectives):
            a = successes[i] + 1
            b = totals - successes[i] + 1
            sample = beta.rvs(a, b, size=1)  # draw from beta distribution
            sampled_means.append(sample[0])
        arm_val = np.array(sampled_means)
        return arm_val

    def ts_sample(self):
        result = []
        for i in range(self.no_arms):
            sampled_val = self.ts_sample_arm(i)
            result.append(sampled_val)
        return result
    
    
class GaussianBanditStats:
    def __init__(self, gt_bandit):
        self.total_count = 0
        self.datapoints = []
        self.counts = []
        self.no_arms = len(gt_bandit.arms)
        self.number_of_objectives = gt_bandit.number_of_objectives
        for i in range(self.no_arms):
            init_estimate = gt_bandit.pull(i)
            singleton = [init_estimate]
            self.datapoints.append(singleton)
            self.counts.append(1.0)
            self.total_count = self.total_count + 1.0

    def enter_data(self, arm, sample):
        self.total_count = self.total_count + 1
        self.counts[arm] = self.counts[arm] + 1
        self.datapoints[arm].append(sample)

    def mo_ucb1_value(self, arm, weight_vector):
        bonus = math.sqrt((2.0 * math.log(self.total_count)) / self.counts[arm])
        arm_val = self.current_estimate(arm)
        scal_val = np.inner(weight_vector, arm_val)
        return scal_val + bonus

    def mo_ucbch_value(self, arm, weight_vector):
        arm_val = self.current_estimate(arm)
        scal_val = np.inner(weight_vector, arm_val)
        bonusa = (scal_val * math.log(self.total_count)) / self.counts[arm]
        if bonusa <= 0.00000000001:
            bonusb = 0
        else:
            bonusb = math.sqrt(bonusa)
        bonusc = math.log(self.total_count) / self.counts[arm]
        bonus = bonusb + bonusc
        return scal_val + bonus

    def current_estimate(self, arm):
        return np.mean(self.datapoints[arm], axis=0)

    def current_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(self.no_arms):
            arm_val = self.current_estimate(arm)
            scal_val = np.inner(weight_vector, arm_val)
            if scal_val >= c_max:
                arm = i
                c_max = scal_val
        return arm, c_max

    def current_ucb1_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(self.no_arms):
            ucb_val = self.mo_ucb1_value(i, weight_vector)
            if ucb_val >= c_max:
                arm = i
                c_max = ucb_val
        return arm, c_max

    def current_ucbch_max(self, weight_vector):
        c_max = 0
        arm = 0
        for i in range(self.no_arms):
            ucb_val = self.mo_ucbch_value(i, weight_vector)
            if ucb_val >= c_max:
                arm = i
                c_max = ucb_val
        return arm, c_max

    def mo_ts_sample_value(self, arm, weight_vector):
        sampled_means = self.ts_sample_arm(arm)
        scal_val = np.array(sampled_means)
        return scal_val

    def ts_sample_max(self, weight_vector):
        #TODO: REPLACE
        mean_max = 0
        arm = 0
        for i in range(self.no_arms):
            sampled_val = self.mo_ts_sample_value(i, weight_vector)
            if sampled_val >= mean_max:
                arm = i
                mean_max = sampled_val
        return arm, mean_max

    def ts_sample_arm(self, arm):
        mu_0 = (1.0/self.number_of_objectives)*np.ones(self.number_of_objectives)
        #print(mu_0)
        #print(np.identity(self.number_of_objectives))
        smpl = self.sample_mean(arm,np.identity(self.number_of_objectives),mu_0,self.number_of_objectives-1,self.number_of_objectives-1)
        return smpl

    def ts_sample(self):
        result = []
        for i in range(self.no_arms):
            sampled_val = self.ts_sample_arm(i)
            result.append(sampled_val)
        return result    
    
    def mean_posterior(self, arm, prior_mu, kappa_0):
       x_bar = self.current_estimate(arm)
       n = len(self.datapoints[arm])
       norm = n + kappa_0
       return ((kappa_0/norm * prior_mu) + (n/norm * x_bar))
   
    def scatter_matrix(self, arm):
        mattie = np.array(self.datapoints[arm])
        #print(mattie)
        #TODO: REPLACE
        Q = (np.cov(mattie,rowvar=False))
        scal = float(len(self.datapoints[arm]) - 1)
        S = scal*Q
        return S
    
    def mu_0_mean_diff(self, arm, mu_0, kappa_0):
        x_bar = self.current_estimate(arm)
        n = len(self.datapoints[arm])
        mmmin = x_bar - mu_0
        mattie = np.dot(np.array([mmmin]).T, np.array([mmmin]))
        scal = kappa_0*n/(kappa_0+n)
        #print("scal: ", str(scal))
        return scal*mattie
    
    def psi_posterior(self, arm, lambda_0, mu_0, kappa_0):
        #print(lambda_0) 
        #print("scatter:")
        #print(self.scatter_matrix(arm))
        #print("m diff:")
        #print(self.mu_0_mean_diff(arm, mu_0, kappa_0))
        #print("sums:")
        #print(lambda_0 + self.scatter_matrix(arm) + self.mu_0_mean_diff(arm, mu_0, kappa_0))
        return lambda_0 + self.scatter_matrix(arm) + self.mu_0_mean_diff(arm, mu_0, kappa_0)
        
    def covariance_posterior(self, arm, lambda_0, mu_0, kappa_0):
        return np.linalg.inv(self.psi_posterior(arm, lambda_0, mu_0, kappa_0))  
    
    def sample_invwishart(self, arm, lambda_0, mu_0, kappa_0, nu_0):
        lambda_n = self.covariance_posterior(arm, lambda_0, mu_0, kappa_0)
        nu_n = float(len(self.datapoints[arm]) + nu_0)
        #print("*** "+str(nu_n))
        #print("*** "+str(lambda_n))
        return scipy.stats.invwishart.rvs(nu_n,lambda_n)
    
    def sample_mean(self, arm, lambda_0, mu_0, kappa_0, nu_0):
        Sigma = self.sample_invwishart(arm, lambda_0, mu_0, kappa_0, nu_0)
        kappa_n = len(self.datapoints[arm]) + kappa_0
        mu_post = self.mean_posterior(arm, mu_0, kappa_0)
        sample = np.random.multivariate_normal(mu_post, (1.0/kappa_n)*Sigma)
        return sample
    
    def significance_test_arms(self, arm1, arm2, p_val):
        D1    = np.array(self.datapoints[arm1])
        D2    = np.array(self.datapoints[arm2])
        #print(D1, D2)
        T2    = spm1d.stats.hotellings2(D1,D2)
        T2i   = T2.inference(p_val)
        return T2i.h0reject
        
        

def sample_weight_vector(w_vec, h_vec):
    if h_vec is None:
        return w_vec
    w_sample = []
    for i in range(len(w_vec)):
        stdev = 1.0 / math.sqrt(h_vec[i])
        # print("stdev "+str(i)+" "+str(stdev))
        ws = np.random.normal(w_vec[i], stdev)
        w_sample.append(ws)
    return w_sample

def sample_gaussian(mu_vec, sig_mat):
    return None

def sample_inverse_wishart(mu_vec, sig_mat):
    return None
