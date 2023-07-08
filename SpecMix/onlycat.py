import numpy as np
from scipy.sparse import spdiags, issparse
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
import warnings
warnings.filterwarnings("ignore")

class onlyCat(BaseEstimator, ClusterMixin):
    '''
    Spectral clustering algorithm with only categorical features.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form as well as the number of
        centroids to generate.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    return_df : bool, default=False
        Whether to return a labeled DataFrame or a numpy array.
    
    Attributes
    ----------
    B : array-like or sparse matrix, shape=(n_samples, n_features)
        Bipartite graph.

    labels_ : array, shape = (n_samples,)
        Cluster labels for each point.
    '''

    def __init__(self, n_clusters=2, random_state=None, return_df=False):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.return_df = return_df
    
    def fit(self, X, y=None):
        '''
        Compute spectral clustering with only categorical features.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
        '''
        # Remove all numerical features
        
        X = X.select_dtypes(exclude=np.number)            
        # Create adjacency matrix
        self.B = pd.get_dummies(X).to_numpy()
        self.labels_ = self.Tcut(self.B)
        return self
    
    def fit_predict(self, X, y=None):
        '''
        Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : array, shape = (n_samples,)
            Cluster labels for each point.
        '''
        self.fit(X)
        return self.labels_
    
        
    def Tcut(self, B):
        """
        Perform spectral clustering on the bipartite graph B, which only contains categorical features.
        Args:
        - B: dataframe/array, the bipartite graph
        Returns:
        - labels: numpy array with shape (num_samples,), the cluster labels
        """

        # B - |X|-by-|Y|, cross-affinity-matrix
        # note that |X| = |Y| + |I|
        B = sparse.csr_matrix(B)

        Nx, Ny = B.shape
        if Ny < self.n_clusters:
            raise ValueError('Need more superpixels!')

        ### build the superpixel graph
        dx = B.sum(axis=1)
        Dx = spdiags(1 / dx.T, 0, Nx, Nx)
        Wy = B.T.dot(Dx).dot(B)

        ### compute Ncut eigenvectors
        # normalized affinity matrix
        d = Wy.sum(axis=1)
        D = spdiags(1.0 / np.sqrt(d.T), 0, Ny, Ny)
        nWy = D.dot(Wy).dot(D)
        nWy = (nWy + nWy.T) / 2

        # compute eigenvectors
        if issparse(nWy):
            evals_large, evecs_large = eigsh(nWy, k=self.n_clusters, which='LM')
        else:
            evals_large, evecs_large = np.linalg.eigsh(nWy)
            evals_large = evals_large[::-1]
            evecs_large = evecs_large[:, ::-1]
        Ncut_evec = D.dot(evecs_large[:, :self.n_clusters])

        ### compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
        evec = Dx.dot(B).dot(Ncut_evec)

        ### k-means clustering
        # normalize each row to unit norm
        evec = evec / (np.sqrt(np.sum(evec ** 2, axis=1, keepdims=True)) + 1e-10)

        # k-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(evec)
        return labels
