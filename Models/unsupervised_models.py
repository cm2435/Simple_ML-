from base_model import BaseModel

class PCA(BaseModel):

    '''
    Class to perform principle component analysis (PCA) on a set of multiple dimentional data in the form
    of numpy arrays

    Parameters
    ------------------
    Number components (int): Number of dimentions to be returned by the PCA algorithm 

    Methods 
    ------------------
    fit- Takes in numpy ndarray of data,
    1) standardises data to mitigate the sensitivity to variances
    2) Compute the covariance matrix between X data dimentions 
    3) Find the Eigenvectors and Eigenvalues of the covariance matrix and sort them
    4) Return the top N dimentions of data with highest variance expressed by largest eigenvector 

    ---
    transform- Takes in numpy ndarray of data,
    Uses saved top N componenets to transform input array to reduce dimentionality 
    '''

    def __init__(self, number_components: int) -> None:
        super().__init__()
        self.number_components = number_components
        self.components = None
        self.mean = None 

    def fit(self, X: np.ndarray) -> None:
        #Mean of X 
        self.mean = np.mean(X, axis=0)
        X = X - self.mean 

        #Covariance of mean to standarise 
        cov = np.cov(X.T)

        #Find Eigenvalues and EigenVectors of cov matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T 

        #Sort matrix eigenvectors 
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idxs], eigenvectors[idxs]

        #Get first n principle components from sorted Eigenvectors 
        self.components = eigenvectors[:self.number_components] 

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X - self.mean 
        return np.dot(X, self.components.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass 




class KNN(BaseModel):
    '''
    Class to classify data based on the class of the K nearest neibour points in the
    np.shape[0] dimentional space of X input data 

    Parameters
    ------------------
    k (int): Number of nearest points in high dimentional space to consider for classification 

    Methods 
    ------------------
    euclidean_distance - Takes in two numpy ndarrays of training data and returns the euclidean distance between them
    ---
    predict- Takes in numpy ndarray of data and:
    1) Calculates euclidean distance to all other points
    2) Finds the K nearest points in the N dimentional space by euclidean metric 
    3) Classifies the data as the most common class of its K nearest neibours- Majority voting.

    '''
    def __init__(self, k: int):
        super().__init__()
        self.k = 5

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        #Find the euclidean distance between two real valued N dimentional vectors
        return np.sqrt(np.sum(x1 - x2) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        #Make a copy of the training data 
        self.X_train, self.y_train = X, y

    def _predict(self, X: np.ndarray) -> int:
        #Calculate the distance between a query point and training data
        distances = [self.euclidean_distance(X, x_train) for x_train in self.X_train]

        #Find the indices of the K nearest neibours of the query/test point 
        k_indices = np.argsort(distances)[:self.k]

        #Find the class labels for the nearest k points, classify based on "majority voting" method
        k_nearest_labels = [self.y_train[index] for index in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        #Predict class labels of testing data 
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels