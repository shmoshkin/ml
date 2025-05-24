import numpy as np
import math

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.0,
            (0, 1): 0.3,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0,
            (0, 1): 0.3,
            (1, 0): 0.5,
            (1, 1): 0.2
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.3,
            (0, 1): 0,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0,
            (0, 0, 1): 0,
            (0, 1, 0): 0,
            (0, 1, 1): 0.3,
            (1, 0, 0): 0.3,
            (1, 0, 1): 0,
            (1, 1, 0): 0.2,
            (1, 1, 1): 0.2,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        p_x = np.array(list(X.values())) # could reshape x to (a,b) and y to (b,a) but not generalizing
        p_y = np.array(list(Y.values())).reshape(2,1) # would generalize dimensions but all binary
        px_py = (p_x * p_y).flatten()
        return not np.allclose(px_py, list(X_Y.values()))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndependent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        trues = 0
        for i,j,k in X_Y_C.keys():
            trues += np.isclose(X_Y_C[i,j,k] / C[k], (X_C[i,k] / C[k] * Y_C[j,k] / C[k]))
        return trues == len(X_Y_C)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    log_p = np.log(((rate ** k) * np.exp(-rate)) / math.factorial(k))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    vectorized_poisson_log_pmf = np.vectorize(poisson_log_pmf)
    log_likelihood_arr = np.array([vectorized_poisson_log_pmf(samples,𝜆) for 𝜆 in rates])
    likelihoods = log_likelihood_arr.sum(axis=1) # sum the rows to get log_lik for each 𝜆
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    rate = rates[np.argmax(likelihoods)] # argmax gives index(row) with highest log_likelihood rate

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean = samples.mean()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.class_value = class_value
        self.input_data = dataset 
        self.dataset = dataset[dataset[:,-1] == self.class_value][:,:-1]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.dataset.shape[0] / self.input_data.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        emp_mean = self.dataset.mean(axis=0)
        emp_std = self.dataset.std(axis=0)
        vect_normal_pdf = np.vectorize(normal_pdf)
        likelihood = np.prod(vect_normal_pdf(x[:-1], emp_mean, emp_std))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pred = np.argmax([self.ccd0.get_instance_posterior(x), self.ccd1.get_instance_posterior(x)])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    prediction = np.apply_along_axis(map_classifier.predict,axis=1,arr=test_set)
    corrects = np.sum(prediction == test_set[:,-1])
    acc = corrects / test_set.shape[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    d_exp, det, inv = (x.shape[0] - 1)/2, np.linalg.det(cov), np.linalg.pinv(cov)
    x_mu = x - mean.T

    first_term = 1 / ((2 * np.pi) ** d_exp * det ** 0.5)
    e_power = -0.5 * (x_mu @ inv @ x_mu)
    pdf = first_term * np.exp(e_power)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.class_value = class_value
        self.input_data = dataset
        self.dataset = dataset[dataset[:,-1] == self.class_value][:, :-1]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.dataset.shape[0] / self.input_data.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        emp_mean = self.dataset.mean(axis=0)
        cov_matrix = np.cov(self.dataset.T)
        likelihood = multi_normal_pdf(x[:-1], emp_mean, cov_matrix)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pred = np.argmax([self.ccd0.get_prior(), self.ccd1.get_prior()])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pred = np.argmax([self.ccd0.get_instance_likelihood(x), self.ccd1.get_instance_likelihood(x)])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.class_value = class_value
        self.input_data = dataset
        self.dataset = dataset[dataset[:,-1] == self.class_value][:, :-1]
        self.n_i = self.dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.n_i / self.input_data.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = 1
        for instance in range (len(x)-1):
            # 𝑛𝑖𝑗 Num of training instances w class 𝐴𝑖 & value 𝑥𝑗 in relevant attribute
            n_ij = self.dataset[self.dataset[:,instance] == x[instance]].shape[0]
            v_j = len(np.unique(self.dataset[:,instance])) # |𝑉𝑗| Num possible vals of relevant attrib
            p_x = ((n_ij + 1) / (self.n_i + np.abs(v_j))) if n_ij > 0 else EPSILLON # value not in support of PMF
            likelihood *= p_x
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc


