## K-Nearest-Neighbors:

### In fit() 
Data with its target, X and y, are given which were initialized in fit() function. 

### In predict()
Actual classification of the data is done. 
First distance of every sample to every sample is found and stored in a list. If the weight of distance is 'distance', invere of distance is stored. Else normal distance is stored.
Then from all the stored distance first n smallest distance are taken into consideration if distance weight is uniform else first n largest distance is taken.
If the distance is uniform, sample is assigned to maximum occuring class, else the distance of all individual distinct class is calculated. Let say, for 5 neighbours (1,2,0,1,1), distance is 1/2+1/3+1/4=1.8 for 1, 1/6 = 1.66 for 0 and 1/7= 1.42. So the sample is addigned to class with max distance which is class 1.

## Multi-Layer Perceptron

## Forward Pass

In the forward pass, weights and input dot product is taken and bias is added to it. Which becomes our hidden layer values. Then, dot product of hidden layed values and weights for output layer is taken and then bias is added. It will be our output.

### Backward Pass

For backward pass, we will take the difference of predicted value and actual output as our error. Then we will multiply error with derivative of output activation which will become our gradient. Then we will take dot product of output and hidden layer multiplying it with learning rate. We will subtract this from our output weights which will update our output weights. Same will be done for the hidden layer weights.
