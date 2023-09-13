from sklearn.cluster import SpectralClustering
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist



class SpecMix(BaseEstimator, ClusterMixin):
    '''
    Spectral clustering with mixed data types.

    Parameters
    ----------
    sigma : float, default=1
        The sigma value for the Gaussian kernel.

    kernel : string, default=None
        The method used to compute the Gaussian kernel. If None, the default is 1. 

    lambdas : list of ints, default=None
        The distance between each pair of categorical variables.

    knn : int, default=0
        The number of nearest neighbors to use for the KNN graph.

    numerical_cols : list of strings, default=[]
        The names of the numerical columns. If empty, automatically determined.

    categorical_cols : list of strings, default=[]
        The names of the categorical columns. If empty, automatically determined.

    n_clusters : int, default=2
        The number of clusters to use for spectral clustering.
    
    scaling : boolean, default=True
        Whether to scale the numerical data or not. 

    sigmas : list of floats, default=[]
        The sigma values to try for cross-validation.
    
    random_state : int, default=0
        Random seed for reproducibility.
    
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
    
    verbose : int, default=0
        Verbosity mode.
    
    return_df : boolean, default=False
        Whether to return a pandas DataFrame or a numpy array.

    Attributes
    ----------
    adj_matrix_ : numpy array with shape (n_samples, n_samples)
        The adjacency matrix used for spectral clustering. Only available after calling the fit() method.
    
    labels_ : numpy array with shape (n_samples,)
        The cluster labels.
    '''

    def __init__(self, sigma = 1, kernel=None, lambdas=None, knn=0,  numerical_cols=[], categorical_cols=[], n_clusters = 2, scaling = True,
                sigmas = [], random_state = 0, n_init = 10, verbose = 0, return_df=False):
        
        self.sigma = sigma
        self.kernel = kernel
        self.lambdas = lambdas
        self.knn = knn
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.n_clusters = n_clusters
        self.scaling = scaling
        self.sigmas = sigmas
        self.random_state = random_state
        self.n_init = n_init
        self.verbose = verbose
        self.return_df = return_df

    def fit(self, X, y=None):
        '''
        Performs spectral clustering on a given dataset.
        Args:
        - X: pandas DataFrame, with shape (n_samples, n_features), the dataset
        - y: Not used, present for API consistency by convention.
        Returns:
        - self
        '''
        self.adj_matrix_ = self.create_adjacency_df(X)
        self.spectral = SpectralClustering(n_clusters=self.n_clusters, assign_labels='kmeans',random_state=self.random_state, affinity = 'precomputed', n_init=self.n_init, verbose=self.verbose).fit(self.adj_matrix_)
        self.labels_ = self.spectral.labels_[:len(X)]
        return self

    def fit_predict(self, X, y=None):
        '''
        Performs spectral clustering on a given dataset and returns the cluster labels.
        Args:
        - X: pandas DataFrame, with shape (n_samples, n_features), the dataset
        - y: Not used, present for API consistency by convention.
        Returns:
        - labels: numpy array with shape (n_samples,), the cluster labels
        '''
        self.fit(X, y)
        return self.labels_



    def create_adjacency_df(self, df): 
        """
        Creates an adjacency matrix for a given dataset for use in spectral clustering.
        Args:
        - df: pandas DataFrame, with shape (num_samples, n_features), the dataset
        Returns:
        - matrix: numpy array/dataframe with shape (num_samples, num_samples), the adjacency matrix
        """

        
        numerical_nodes_count = len(df.index)
        df = df.drop(['target'], axis=1, errors='ignore')

        numerical_labels = []  # keep track of numerical node labels
        categorical_labels = []  # keep track of categorical node labels
        # If columns are not specified, determine them automatically
        if not self.numerical_cols and not self.categorical_cols: 
            # Separate numeric and categorical columns
            numeric_df = df.select_dtypes(include=np.number)
            categorical_df = df.select_dtypes(exclude=np.number)
        else:
            numeric_df = df[self.numerical_cols]
            categorical_df = df[self.categorical_cols]

        if not self.lambdas:
            self.lambdas = [1] * len(categorical_df.columns)

        # Add numerical labels to list
        for i in range(numerical_nodes_count):
            numerical_labels.append(f'numerical{i}')

        # Add categorical labels to list
        for k, col in enumerate(categorical_df):
            for value in categorical_df[col].unique():
                categorical_labels.append(f'{col}={value}')
        categorical_nodes_count = len(categorical_labels)
        total_nodes_count = numerical_nodes_count + categorical_nodes_count

        # Initialize adjacency matrix
        matrix = np.zeros((total_nodes_count, total_nodes_count))

        # Calculate numerical distances using KNN graph or fully connected graph
        if not numeric_df.empty:
            # Scale numerical data
            if self.scaling:
                scaler = StandardScaler()
                numeric_arr = scaler.fit_transform(np.array(numeric_df))
            else:
                numeric_arr = np.array(numeric_df)
            if self.kernel:
                if self.kernel == "median_pairwise":
                    self.sigma = self.median_pairwise(numeric_arr)
                elif self.kernel == "cv_sigma":
                    self.sigma = self.cv_sigma(numeric_arr)
                elif self.kernel == "auto":
                    # Calculate the standard deviation of the distances
                    self.sigma = self.std_weights(numeric_arr)
                elif self.kernel == "preset":
                    pass
                else:
                    raise ValueError("Invalid kernel value. Must be one of: median_pairwise, ascmsd, cv_distortion, cv_sigma")
            # Avoid division by zero
            if self.sigma == 0:
                self.sigma = 1e-10
            # Compute the distance matrix using KNN graph or fully connected graph
            if self.knn:
                A_dist = kneighbors_graph(numeric_arr, n_neighbors=self.knn, mode='distance', include_self=True)
                A_conn = kneighbors_graph(numeric_arr, n_neighbors=self.knn, mode='connectivity', include_self=True)
                A_dist = A_dist.toarray()
                A_conn = A_conn.toarray()

                # Make connectivity and distance matrices symmetric
                A_conn = 0.5 * (A_conn + A_conn.T)
                A_dist = 0.5 * (A_dist + A_dist.T)

                # Compute the similarities using boolean indexing
                dist_matrix = np.exp(-(A_dist)**2 / ((2 * self.sigma**2)))
                dist_matrix[~A_conn.astype(bool)] = 0.0
            else:
                dist_matrix = cdist(numeric_arr, numeric_arr, metric='euclidean')
                dist_matrix = np.exp(-(dist_matrix)**2 / ((2 * self.sigma**2)))

            # Return distance matrix if there are no categorical features
            if self.lambdas and self.lambdas[0] == 0:
                return dist_matrix if not self.return_df else (pd.DataFrame(dist_matrix, index=numerical_labels, columns=numerical_labels))
            
            # Add numerical distance matrix to the original one (top left corner)
            matrix[:numerical_nodes_count, :numerical_nodes_count] = dist_matrix

        # Connect categorical nodes to numerical observations
        for i in range(numerical_nodes_count):
            for k, col in enumerate(categorical_df):
                j = numerical_nodes_count + categorical_labels.index(f'{col}={categorical_df[col][i]}')
                matrix[i][j], matrix[j][i] = self.lambdas[k], self.lambdas[k]
        # Create labeled DataFrame if required
        if self.return_df:
            return pd.DataFrame(matrix, index=numerical_labels + categorical_labels, columns=numerical_labels + categorical_labels)
        else:
            return matrix
        

    def median_pairwise(self, numeric_arr):
        '''
        Computes the median pairwise distance between all points in a dataset.
        Args:
        - numeric_arr: numpy array with shape (num_samples, n_features), the dataset
        Returns:
        - sigma: float, the median pairwise distance
        '''    
        sigma = np.median(numeric_arr)
        return sigma


    def cv_sigma(self, adjacency_matrix, scoring_function=silhouette_score):
        """
        Performs cross-validation to find the best sigma value for the Gaussian kernel.
        Args:
        - adjacency_matrix: numpy array with shape (num_samples, num_samples), the adjacency matrix
        - scoring_function: function, the scoring function to use for cross-validation
        Returns:
        - best_sigma: float, the best sigma value
        """
        best_sigma = None
        best_score = -np.inf

        for sigma in self.sigmas:
            # Apply spectral clustering with the current sigma
            sc = SpectralClustering(n_clusters=self.n_clusters, affinity='rbf', gamma=1.0/sigma**2)
            cluster_labels = sc.fit_predict(adjacency_matrix)
            # Compute score
            if len(np.unique(cluster_labels)) == 1:
                score = -np.inf
            else:
                score = scoring_function(adjacency_matrix, cluster_labels)

            # Update the best score and best sigma if current score is higher
            if score > best_score:
                best_score = score
                best_sigma = sigma

        return best_sigma

    def std_weights(self, adjacency_matrix):
        '''
        Computes the standard deviation of the pairwise distances between all points in a dataset, to find
        best sigma value for the Gaussian kernel.
        Args:
            - adjacency_matrix: numpy array with shape (num_samples, num_samples), the adjacency matrix
        Returns:
        - sigma: float, the standard deviation of the pairwise distances
        '''
        sigma = np.std(cdist(adjacency_matrix, adjacency_matrix, metric='euclidean'))
        return sigma