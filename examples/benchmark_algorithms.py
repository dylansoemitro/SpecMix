from sklearn.metrics import jaccard_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score, homogeneity_score, confusion_matrix
from itertools import permutations
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from examples.spectralCAT import spectralCAT
from SpecMix.onlycat import OnlyCat
from sklearn.cluster import KMeans
from prince import FAMD
import numpy as np
import time
import gower
from SpecMix.specmix import SpecMix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def calculate_score(df, target_labels, n_clusters = 2, method = "specmix", metrics = ["jaccard"], sigma=1, 
                    kernel = None, lambdas=[], knn=0, binary_cols = [], categorical_cols = [], numerical_cols = [],  
                    scaling = True, sigmas = [], random_state = 0, n_init = 10, verbose = 0):
  """
  Calculates the score of a clustering algorithm on a dataset
  Parameters
  ----------
  df : pandas dataframe
      Dataframe containing the dataset
  target_labels : list
      List containing the true labels of the dataset
  n_clusters : int, optional
      Number of clusters to form. The default is 2.
  method : string, optional
      Clustering algorithm to use. The default is "specmix".
  metrics : list, optional
      List of metrics to use for calculating the score. The default is ["jaccard"].
  sigma : float, optional
      Sigma value for the Gaussian kernel. The default is 1.
  kernel : string, optional
      Kernel to use for the Gaussian kernel. The default is None (preset to the sigma value given).
  lambdas : list, optional
      List of lambda values for the Gaussian kernel, the distance between each pair of categorical variables. The default is [].
  knn : int, optional
      Number of nearest neighbors to use for the Gaussian kernel. The default is 0.
  binary_cols : list, optional
      List of binary columns in the dataset. The default is [].
  categorical_cols : list, optional
      List of categorical columns in the dataset. The default is [].
  numerical_cols : list, optional
      List of numerical columns in the dataset. The default is [].
  scaling : bool, optional
      Whether to scale the data. The default is True.
  sigmas : list, optional
      List of sigma values for the Gaussian kernel. The default is [].
  random_state : int, optional
      Random state to use for the clustering algorithm. The default is 0.
  n_init : int, optional
      Number of times the k-means algorithm will be run with different centroid seeds. The default is 10.
  verbose : int, optional
      Verbosity mode. The default is 0.
  Returns
  -------
  scores_dict : dict
      Dictionary containing the scores for each metric.
  time_taken : float
      Time taken for the clustering algorithm to run.
  """
  
  # Drop target column if present
  df = df.drop(['target'], axis=1, errors='ignore')

  #Check that total number of columns is equal to the sum of the number of numerical, categorical and binary columns
  if numerical_cols and categorical_cols and binary_cols and not len(numerical_cols) + len(categorical_cols) + len(binary_cols) == df.shape[1]:
    raise ValueError("Number of columns in numerical, categorical and binary columns lists should be equal to total number of columns in dataframe")

  # Convert columns to appropriate data types
  le = LabelEncoder()

  for col in categorical_cols:
      df[col] = df[col].astype('object')
      df[col] = le.fit_transform(df[col])
      df[col] = df[col].astype('object')

  for col in numerical_cols:
    df[col] = df[col].astype('float')
  for col in binary_cols:
    df[col] = df[col].astype('bool')
  if method == "specmix":
    # Throw error if only categorical columns are present
    if not numerical_cols and categorical_cols:
      raise ValueError("Only categorical columns are present. Please use OnlyCat instead.")
    specmix = SpecMix(n_clusters=n_clusters, sigma=sigma, kernel=kernel, lambdas=lambdas, knn=knn, numerical_cols=numerical_cols, categorical_cols=categorical_cols + binary_cols,
                       scaling=scaling, sigmas=sigmas, random_state=random_state, n_init=n_init, verbose=verbose)
    start_time = time.time()
    specmix.fit(df)
    end_time = time.time()
    predicted_labels = specmix.labels_

  elif method == "k-prototypes":
    if categorical_cols:
        catColumnsPos = [df.columns.get_loc(col) for col in categorical_cols + binary_cols]
    else:
        catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
    if catColumnsPos and not len(numerical_cols) == df.shape[1]:
      kprototypes = KPrototypes(n_jobs = -1, n_clusters = n_clusters, init = 'Huang', random_state = 0)
      start_time = time.time()
      predicted_labels = kprototypes.fit_predict(df.to_numpy(), categorical = catColumnsPos)
      end_time = time.time()

  elif method == "k-means":
      #Throw error if categorical columns are present
      if categorical_cols:   
        raise ValueError("Categorical columns are not supported for k-means")
      start_time = time.time()
      kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, verbose=verbose).fit(df.to_numpy())
      end_time = time.time()
      predicted_labels = kmeans.labels_
    
  elif method == "k-modes":
    #Throw error if numerical columns are present
    if numerical_cols:
      raise ValueError("Numerical columns are not supported for k-modes")
    k_modes = KModes(n_clusters=n_clusters, init='Huang',  random_state=random_state, n_init=n_init, verbose=verbose)
    start_time = time.time()
    predicted_labels = k_modes.fit_predict(df)
    end_time = time.time()

  elif method == "lca":
    # Extract binary columns
    if not categorical_cols:
      # Extract numerical columns
      numCols= set(df.select_dtypes(include=[int, float]).columns)
      # Extract categorical columns
      binCols = set(df.select_dtypes(include=[bool]).columns)
      catCols = set(df.columns) - numCols - binCols
    else:
      catCols = set(categorical_cols)
      numCols = set(numerical_cols)
      binCols = set(binary_cols)

    #Purely continuous data
    if not catCols and not binary_cols:
      model = StepMix(n_components=n_clusters, measurement="continuous", verbose=verbose, random_state=random_state)
      mixed_data = df
    else:
      mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=df,
        continuous=numCols,
        binary=binCols,
        categorical=catCols)
      model = StepMix(n_components=n_clusters, measurement=mixed_descriptor, verbose=verbose, random_state=random_state)
    # Fit model
    start_time = time.time()
    model.fit(mixed_data)
    end_time = time.time()
    # Class predictions
    df['mixed_pred'] = model.predict(mixed_data)
    predicted_labels = df['mixed_pred'].to_numpy()
    end_time = time.time()

  elif method == "spectralCAT":
    start_time = time.time()
    predicted_labels= spectralCAT(df.copy(), n_clusters, random_state, 0)
    end_time = time.time()

  elif method == "famd":
    famd = FAMD(n_components=n_clusters, random_state=random_state, copy=True)
    start_time = time.time()
    famd.fit(df)
    transformed = famd.transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, verbose=verbose).fit(transformed)
    predicted_labels = kmeans.predict(transformed)
    end_time = time.time()

  elif method == "onlycat":
    df = df.drop(['target'], axis=1, errors='ignore')
    start_time = time.time()
    predicted_labels = OnlyCat(n_clusters=n_clusters, random_state=random_state).fit_predict(df)
    end_time = time.time()
  else:
    raise ValueError("Invalid method")
  time_taken = end_time - start_time
  scores_dict = {}
  # element_frequencies_pred = Counter(predicted_labels)
  # element_frequencies_target = Counter(target_labels)
  # print("Predicted Label Frequencies: ")
  # for element, frequency in element_frequencies_pred.items():
  #     print(f"Element {element}: {frequency} times")
  # print("Target Label Frequencies: ")
  # for element, frequency in element_frequencies_target.items():
  #     print(f"Element {element}: {frequency} times")

  score_function_dict = {"jaccard": jaccard_score, "purity": purity_score, "silhouette": silhouette_score, 
                         "calinski_harabasz": calinski_harabasz_score, "adjusted_rand": adjusted_rand_score, 
                         "homogeneity": homogeneity_score}
  for metric in metrics:
    scores_list = []
    score_function = score_function_dict[metric]

    if score_function == jaccard_score:
      #For jaccard score, we need to find the permutation of predicted labels that maximizes the score
      for perm in permutations(range(n_clusters)):
          perm_predicted_labels = [perm[label] for label in predicted_labels]
          if n_clusters > 2:
            score = score_function(perm_predicted_labels, target_labels, average='weighted')
          else:
            score = score_function(perm_predicted_labels, target_labels)
          scores_list.append(score)
      score = max(scores_list)

    elif score_function == silhouette_score or score_function == calinski_harabasz_score:
      gower_dist_matrix = gower.gower_matrix(df)      
      # score is -1 if all points are in one cluster
      if len(np.unique(predicted_labels)) == 1:
        score = -1
      else:
        if score_function == silhouette_score:
          score = score_function(gower_dist_matrix, predicted_labels, metric = 'precomputed')
        else:
          score = score_function(gower_dist_matrix, predicted_labels)
    else:
      score = score_function(predicted_labels, target_labels)
    scores_dict[metric] = score
  return scores_dict, time_taken


def purity_score(y_pred, y_true):
    """Purity score
        To compute purity, each cluster is assigned to the class which is most frequent
        in the cluster [1], and then the accuracy of this assignment is measured by counting
        the number of correctly assigned data points and dividing by the total number of data points.
        We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
        the clusters index.
        Args:
            y_pred(np.ndarray): the predicted clusters
            y_true(np.ndarray): the true labels
        Returns:
            float: the purity score
        References:
            [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find the maximum value in each column (majority class)
    majority_sum = np.sum(np.amax(cm, axis=0))
    # Calculate purity
    purity = majority_sum / np.sum(cm)
    
    return purity

def compare_algorithms(methods, df, target_labels, n_clusters = 2, metrics = ["jaccard"], sigma=1, 
                    kernels = [] , lambda_values=[], knn=0, binary_cols = [], categorical_cols = [], numerical_cols = [],  
                    scaling = True, sigmas = [], random_state = 0, n_init = 10, verbose = 0):
  """
  Compares the scores of multiple clustering algorithms on a dataset
  Parameters
  ----------
  methods : list
      List of clustering algorithms to compare.
  df : pandas dataframe
      Dataframe containing the dataset
  target_labels : list
      List containing the true labels of the dataset
  n_clusters : int, optional
      Number of clusters to form. The default is 2.
  metrics : list, optional
      List of metrics to use for calculating the score. The default is ["jaccard"].
  sigma : float, optional
      Sigma value for the Gaussian kernel. The default is 1.
  kernels: list, optional
      List of kernels to use for the Gaussian kernel. The default is [] (preset to the sigma value given).
  lambdas : list, optional
      List of lambda values for the Gaussian kernel, the distance between each pair of categorical variables. The default is [].
  knn : int, optional
      Number of nearest neighbors to
        use for the Gaussian kernel. The default is 0.
  binary_cols : list, optional
      List of binary columns in the dataset. The default is [].
  categorical_cols : list, optional
      List of categorical columns in the dataset. The default is [].
  numerical_cols : list, optional
      List of numerical columns in the dataset. The default is [].
  scaling : bool, optional
      Whether to scale the data. The default is True.
  sigmas : list, optional
      List of sigma values for the Gaussian kernel. The default is [].
  random_state : int, optional
      Random state to use for the clustering algorithm. The default is 0.
  n_init : int, optional
      Number of times the k-means algorithm will be run with different centroid seeds. The default is 10.
  verbose : int, optional
      Verbosity mode. The default is 0.
  Returns
  -------
  scores_df : pandas dataframe
      Dataframe containing the scores for each metric for each clustering algorithm.
  """

  # Check for numerical and categorical columns, automatically detect if not specified
  if not numerical_cols and not categorical_cols:
    numerical_cols = list(df.select_dtypes(include=[int, float]).columns)
    categorical_cols = list(df.select_dtypes(include=[object, bool]).columns)
  
  # Remove target column from categorical columns or numerical columns if present
  if 'target' in categorical_cols:
    categorical_cols.remove('target')
  if 'target' in numerical_cols:
    numerical_cols.remove('target')
  

  scores_dict = {method: {} for method in methods}
  time_taken_dict = {method: {} for method in methods}
  
  for method in methods:
    if method == "specmix":
      continue
    scores_dict[method], time_taken_dict[method] = calculate_score(df, target_labels, n_clusters, method, metrics, sigma, 
                    kernels, lambda_values, knn, binary_cols, categorical_cols, numerical_cols,  
                    scaling, sigmas, random_state, n_init, verbose)
  # Calculate scores for SpecMix
  if "specmix" in methods:
    for ker in kernels:
      for l in lambda_values:
        scores_dict[f'specmix lambda={l} kernel={ker}'], time_taken_dict[f'specmix lambda={l} kernel={ker}'] = calculate_score(df, target_labels, n_clusters, "specmix", metrics, sigma, 
                      ker, [l] * len(categorical_cols), knn, binary_cols, categorical_cols, numerical_cols,  
                      scaling, sigmas, random_state, n_init, verbose)
  scores_dict.pop("specmix")
  time_taken_dict.pop("specmix")

  scores_df = pd.DataFrame(scores_dict)
  time_taken_df = pd.DataFrame(time_taken_dict, index = ['time_taken'])
  scores_df = pd.concat([scores_df, time_taken_df])
  return scores_df

