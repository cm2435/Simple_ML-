from base_model import BaseModel

class Gaussian_Naive_Bayes(BaseModel):
    '''
    Classifier class that finds the conditional proability of y given input data X
    using bayes rule 

    Parameters
    ------------------
    classes (None type)- classes of input data to be predicted by classifier
    
    Methods 
    ------------------
    fit- Takes in numpy ndarray of data and for each class 
    Calculates the mean, variance and prior distributions
  
    ---
    pdf- Takes in numpy ndarray of data and for a given class uses the means and variances
    to calculate the probability density function based around a gaussian distribution 

    ---
    predict- Takes in numpy ndarray and calculates the posterior distribution for each class,
    returns a list of classes that are most likely by size of probability
    '''

    def __init__(self):
        super().__init__()
        self._classes = None
        self._mean = {}
        self._variance = {}
        self._priors = {}


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self._classes = np.unique(y)

        #Calculate the mean, variance and priors for the fitting data 
        for idx, c in enumerate(self._classes):
            X_c = X[c==y]
            self._mean[idx] = X_c.mean(axis=0)
            self._variance[idx] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / n_samples


    def _pdf(self, idx: int, x: np.ndarray) -> float:
        #Use a gaussian distribution to find class conditional probability
        mean = self._mean[idx]
        variance = self._variance[idx]
        numerator = np.exp(-(x - mean) ** 2/(2 * variance))
        denominator = np.sqrt(2* np.pi * variance)
        return numerator / denominator


    def _predict(self, X: np.ndarray) -> np.ndarray:
        #Use fitted values for priors to calculate to probability of all classes from posteriors
        posteriors =  [np.sum(np.log(self._pdf(idx, x))) + np.log(self._priors[idx])
                       for idx, c in enumerate(self._classes)]
        return self._classes[np.argmax(posteriors)]

        
    def predict(self, X: np.ndarray) -> np.ndarray:
        #Predict the classes of the test data
        return np.array(self._predict(x) for x in X)




class LinearRegression(BaseModel):
    '''
    Implimentation of linear regression, takes in a N dimentional np.ndarray of training
    data and fits weights and biases to predict values for the independant variable, minimising
    mean squared error. 

    Parameters
    ------------------
    lr (float)- Update rate for the weights and biases, higher lr converge faster with more numeric volatility
    n_iters (float): Number of iterations of gradient descent performed to minimise loss 

    
    Methods 
    ------------------
    fit- Takes in numpy ndarray of data, initialises unit matrix of weights and biases and performs 
    vectorised descent to fit weights and biases 
  
    ---
    predict- Takes in numpy ndarray and makes a prediction of the value of the independant variable
    with the fitted weights and biases
    '''
    def __init__(self, lr: float = 0.02, n_iters: float = 500):
        super().__init__()
        self.n_iters = n_iters
        self.lr = lr 
        self.theta = None 

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  

        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape

        #Initialise the weights and biases for fitting
        self.theta = np.ones(n_features)

        #Impliment vectorised gradient descent to minimise loss
        for _ in range(self.n_iters):
            self.theta -= self.lr * \
                ((1/ n_samples) * X.T @ (X @ self.theta - y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        #Insert the X0 column with 1's to perform scalar multiplication with biases
        X = np.insert(0, 1, axis=1)

        #return predictions on test data
        return X @ self.theta




class LogisticRegression(BaseModel):
    '''
    Implimentation of logistic regression for binary classification, takes in a N dimentional np.ndarray
    of training data and uses the logistic function to make a prediction between 0-1, if y^ is > a threshold
    predict in a specific class

    Parameters
    ------------------
    lr (float)- Update rate for the weights and biases, higher lr converge faster with more numeric volatility
    n_iters (float): Number of iterations of gradient descent performed to minimise loss 

    
    Methods 
    ------------------
    fit- Takes in numpy ndarray of data, initialises unit matrix of weights and biases and performs 
    vectorised descent passed through sigmoid function to minimise loss 
  
    ---
    predict- Takes in numpy ndarray and makes a prediction of the value of the independant variable
    with the fitted weights and biases, if greater than 0.5 predict clas 1, else class 0.
    '''
    def __init__(self, lr: float = 0.1, n_iters: int = 100):
        super().__init__()
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None 
    
    def sigmoid(self, Z: nd.ndarray) -> np.ndarray:
        #Takes in array of len(Z) and passes each element through sigmoid function 
        return 1 / (1 + np.exp(-Z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: 

        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape

        #Initialise the weights and biases for fitting
        self.theta = np.ones(n_features)

        #Impliment vectorised gradient descent using logistic regression function 
        for _ in range(self.n_iters):
            self.theta -= self.lr * (1/ n_samples) * \
                X.T @ (self.sigmoid(X @ self.theta) - y)


    def predict(self, X: nd.ndarray) -> np.ndarray:
        #Insert the X0 column with 1's to perform scalar multiplication with biases
        X = np.insert(0, 1, axis=1) 

        #Return predictions on clas test data based on side of prediction threshold of 0.5
        return np.where(self.sigmoid(X @ self.theta) > 0.5, 1, 0)


    
