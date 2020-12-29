##### CREATE LOGISTIC REGRESSION MODEL W/O ML SPECIFIC MODULE #####
import numpy as np

class LogisticRegression:

    # Define attributes and methods of LogisticRegression class
    def __init__(self, learningrate = 0.001, nIter = 1000): # defines learning rate and iteration number for gradient descent
        self.learningrate = learningrate
        self.nIter = nIter
        self.weights = None
        self.bias = None
    
    def gradientDescent(self, X, y): # Where X is a numpy vector (number of samples by number of features per sample)
        # y is a 1-d vector len m (same size as number of samples)
        # initiate the parameters to be used
        numOfSamples, numOfFeatures = X.shape # This takes the first Dim of x and puts it in numOfSamples & second Dim into numOfFeatures
        self.weights = np.zeros(numOfFeatures) # Create a vector of zeroes thats same length as features
        self.bias = 0 # Initiating bias at 0 for simplicity

        # Begin gradient descent loop
        for i in range(self.nIter): # For 1000 iterations
            # Now use sigmoid equation to predict yhat (h(x)): 1/(1 + e^-(weight*input)+bias)
            predicted_y = (1) / (1 + np.exp(-(np.dot(X, self.weights) + self.bias)))

            # Have to update weight after prediction, see YT LogisticRegression 6.5 for minimizing J(theta)
            # Have to use X vector so all samples are updated at the same time
            dw = (1 / numOfSamples) * np.dot(X.T, (predicted_y - y)) 
            db = (1 / numOfSamples) * np.sum(predicted_y - y)

            # Update the weight variable and bias at end of iter step
            self.weights -= self.learningrate * dw
            self.bias -= self.learningrate*db


    def predict(self, X):  # Create function to obtain new test samples to predict
        predicted_y = (1) / (1 + np.exp(-(np.dot(X, self.weights) + self.bias))) # provides probability from sigmoid function using weights and bias trained from gradient descent
        
        # Create the classification logic, if probability >= 0.5, round to 1
        classificationOfY = [1 if i > 0.5 else 0 for i in predicted_y]
        # This for loor does not return a vector as intended, list comprehension work though
        # for i in predicted_y:
        #     if i >= 5: 
        #         classificationOfY = 1
        #     else: 
        #         classificationOfY = 0
        return classificationOfY


    






