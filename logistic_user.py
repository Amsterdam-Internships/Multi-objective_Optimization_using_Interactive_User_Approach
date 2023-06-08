import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from sklearn import linear_model
from momabs_bayesian import bayes_logistic as bl
import random
from gp_pref_elicit_luisa.gp_utilities import utils_user as gp_utils_users
from scipy.stats import multivariate_normal


class LogisticDecisionMaker:
    def __init__(self, no_obj, num_features, noise=0.0, predefined_weights=None, defer_comp_to=None):
        self.weights = np.zeros(num_features)
        self.irrationality_sigma = noise
        self.previous_comparisons = []
        self.previous_outcomes = []
        self.user_pref = gp_utils_users.UserPreference(num_objectives=no_obj, std_noise=0.1)
        self.defer_to_DM = defer_comp_to
        if predefined_weights is not None:
            self.weights = predefined_weights
        else:
            new_seed = math.floor(random.random() * 1000)
            random.seed(new_seed)
            sum_w = 0.0
            for j in range(num_features):
                rnd = random.random()
                self.weights[j] = rnd
                sum_w = sum_w + rnd
            for j in range(num_features):
                self.weights[j] = self.weights[j] / sum_w
                # print("DM LINWEIGHTS: ", self.weights)

    def ground_utility(self, vector): # gives a ground truth utility for the user
        return self.user_pref.get_preference(vector, add_noise=True)

    def features(self, v): # converts input features to feature vectors and gives an array of feature vectors
        mins = []
        maxs = []
        for i in range(len(v)-1):
            for v_j in np.asarray(v[i+1:]).tolist():
                mins.append(min(v[i], v_j))
                maxs.append(max(v[i], v_j))
        result = list(v) + maxs + mins
        return np.array(result)

    def log_comparison(self, vector1, vector2, comp): # just logs the comparison
        v_1 = self.features(vector1)
        v_2 = self.features(vector2)
        # diff_vector = self.features(vector1) - self.features(vector2)
        diff_vector = v_1 - v_2
        res = 0.0
        if comp:
            res = 1.0
        self.previous_comparisons.append(diff_vector)
        self.previous_outcomes.append(res)

    def exact_compare(self, vector1, vector2, log=True): #updates the bayesian model 
        scalar1 = self.ground_utility(vector1)
        scalar2 = self.ground_utility(vector2)
        comp = (scalar1 >= scalar2)
        if log:
            self.log_comparison(vector1, vector2, comp)
        return vector1 if comp else vector2

    def current_map(self): # gives the output for BLR which is the mean and the covariance matrix
        # TODO: H_prior should be made as a matrix (not imp)
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
            return w_fit, H_fit
        else:
            result = np.ones(len(self.weights))
            for i in range(len(self.weights)):
                result[i] = result[i] / float(len(self.weights))
            return result, None
        
    def sample_model(self): # generates sampling from a multivariate gaussian distribution
        w, H = self.current_map()
        norm_dist = multivariate_normal(mean=w, cov=H, allow_singular=False, seed=None)
        w_s = norm_dist.rvs()
        return w_s
    
    def thompson_sampled_point(self, dataset): # gets current best point from BLR according to thompson sampling
        dataset_features = [self.features(v) for v in dataset]
        w_sample = self.sample_model()
        utilities = [np.inner(w_sample, v) for v in dataset_features]
        utilities_max = np.argmax(utilities)
        current_best = dataset[utilities_max]
        return current_best
    
    # TODO: method for excluding points 

        
        

        
