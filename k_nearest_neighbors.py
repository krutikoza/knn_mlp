# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff
import sys
import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """
    
    

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

        #raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        
 
 

        # get distance
        predicted_val = []
        predicted_val_index = []
        store_distance = []
        temp = []
        
        for i in X:
            for j in self._X:
                if self.weights == 'distance':
                    if self._distance(i,j) == 0.0:
                        temp.append(sys.maxsize)  
                    else:
                        temp.append(1/self._distance(i,j))
                    
                else:
                    temp.append(self._distance(i,j))
            store_distance.append(temp)
            temp = []
            
        
        
        
        first_n_distances = []
        
        
        # get first n min distances
        for i in range(len(X)):
            temp = store_distance
            temp_sorted_index = np.argsort(store_distance[i])
            
            temp = []
            temp2 = []
            if self.weights == 'distance':
                temp_sorted_index = temp_sorted_index[::-1]
                for j in range(self.n_neighbors):
                    temp.append(temp_sorted_index[j])
                    temp2.append(store_distance[i][j])
            else:
                for j in range(self.n_neighbors):
                    temp.append(temp_sorted_index[j])
                    temp2.append(store_distance[i][j])

            predicted_val_index.append(temp)
            first_n_distances.append(temp2)
        
        
        
        # find the class in which our test data belong
        if self.weights == 'uniform':
            for i in range(len(X)):
                temp = []
                for j in predicted_val_index[i]:
                    temp.append(self._y[j])
                predicted_val.append(max(temp,key=temp.count))
            return predicted_val
            
        
        else:
            final_y_value = []
            for i in range(len(X)):
                temp = []
                temp_y_value = 0
                
                for j in predicted_val_index[i]:
                    temp.append(self._y[j])
                    
                min_val = -sys.maxsize
                for k in np.unique(temp):
                    sum = 0
                    
                    for j in range(len(temp)):
                        if k == temp[j]:
                            sum = sum + first_n_distances[i][j]
                    if sum > min_val:
                        temp_y_value = k
                        min_val = sum
                
                final_y_value.append(temp_y_value)
                
        return final_y_value
    


        # raise NotImplementedError('This function must be implemented by the student.')