###### Your ID ######
# ID1: 207640822
# ID2: 
#####################

# imports 
import numpy as np
import pandas as pd

def mean_normalize(data):
    return (data - data.mean(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X, y = mean_normalize(X), mean_normalize(y)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    return np.column_stack((np.ones(X.shape[0]), X))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    J = np.mean((X @ theta - y) ** 2) / 2 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    m = X.shape[0]

    for gd in range(num_iters): 
        J_history.append(compute_cost(X, y, theta)) 
        theta -= (alpha / m) * X.T @ (X @ theta - y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    pinv_theta = np.linalg.inv(X.T @ X) @ X.T @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    m = X.shape[0] # m instances - needed for GD formula (unless use .mean)
    threshold = 1e-8
    loss_chg = 1
    iters = 1
    
    while loss_chg >= threshold:
        J_history.append(compute_cost(X, y, theta))    
        theta -= (alpha / m) * X.T @ (X @ theta - y)
        
        if iters > 1: 
            loss_chg = abs(J_history[iters - 1] - J_history[iters - 2]) # account for diff indexes
        iters += 1 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    for key in alphas:
        np.random.seed(42)
        theta = np.random.random(X_train.shape[1])

        theta, J_hist = gradient_descent(X_train, y_train, theta, key, iterations)

        alpha_dict[key] = compute_cost(X_val, y_val, theta)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    remaining_features = list(range(X_train.shape[1]))
    for _ in range(5):  # We want to select 5 features
        best_feature = None
        lowest_val_loss = float('inf')

        for feature in remaining_features:
            # Try adding this feature to the current selection
            trial_features = selected_features + [feature]

            # Prepare training and validation sets with bias
            X_train_sub = np.column_stack((np.ones(X_train.shape[0]), X_train[:, trial_features]))
            X_val_sub = np.column_stack((np.ones(X_val.shape[0]), X_val[:, trial_features]))

            # Initialize theta for gradient descent
            theta_init = np.zeros(X_train_sub.shape[1])

            # Train model using efficient gradient descent
            theta, _ = efficient_gradient_descent(X_train_sub, y_train, theta_init, best_alpha, iterations)

            # Compute validation loss
            val_loss = compute_cost(X_val_sub, y_val, theta)

            # Update best feature if this one is better
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_feature = feature

        # Update selected and remaining features
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    for col in df.columns:
        df_poly[f"{col}_squared"] = df[col] ** 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly