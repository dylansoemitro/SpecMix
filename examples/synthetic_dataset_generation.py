import pandas as pd
import numpy as np


def generate_mixed_dataset(n_numerical_features=2, n_categorical_features=1, n_samples=100, n_clusters=2, mu_range=(-1,1), sigma_range = (0.1,0.3), p=0, precomputed_centers = False, precomputed_sigma = 0.2, seed=0, save=False, filename="mixed_dataset.csv"):
    """
    Generates a mixed type dataset with numerical and categorical features and k clusters.

    Args:
    - n_numerical_features: int, the number of numerical features in the dataset
    - n_categorical_features: int, the number of categorical features in the dataset
    - n_samples: int, the number of samples in the dataset
    - n_clusters: int, the number of clusters to generate
    - mu_range: tuple (float, float), the range of values for the mean of the numerical features
    - sigma_range: tuple (float, float), the range of values for the standard deviation of the numerical features
    - p: float, the noise value for the categorical features
    - precomputed_centers: boolean, whether to use precomputed centers for the clusters
    - precomputed_sigma: float, the sigma value to use for the Gaussian distribution
    - seed: int, random seed for reproducibility
    - save: boolean, whether to save the dataset as a csv file
    - filename: string, the name of the csv file to save the dataset
    Returns:
    - df: pandas DataFrame with shape (n_samples, n_numerical_features + n_categorical_features + 1), the features and the target label of the dataset
    """

    np.random.seed(seed)

    # Situation where we want to generate k clusters with precomputed centers
    if precomputed_centers:
      mu = np.eye(n_clusters)
      sigma = np.full((n_clusters, n_clusters), precomputed_sigma)

    #Generate the numerical features based on Gaussian distributions
    else:
      mu = np.random.uniform(mu_range[0], mu_range[1], size=(n_clusters, n_numerical_features))
      sigma = np.random.uniform(sigma_range[0], sigma_range[1], size=(n_clusters, n_numerical_features))
    cluster_idx = np.random.randint(n_clusters, size=n_samples)
    cluster_idx = np.sort(cluster_idx)
    X_num = np.zeros((n_samples, n_numerical_features))
    for i in range(n_clusters):
        idx = np.where(cluster_idx == i)[0]
        X_num[idx] = np.random.normal(mu[i], sigma[i], size=(len(idx), n_numerical_features))

    # Generate the categorical features
    categories = []
    for i in range(n_clusters):
        cluster_categories = np.random.choice(10, size=n_categorical_features, replace=False)
        categories.append(cluster_categories)
        
    X_cat = np.zeros((n_samples, n_categorical_features))
    X_cat = np.tile(cluster_idx, (n_categorical_features,1))

    mask = np.random.choice([True, False], size=(n_categorical_features, n_samples), p=[p, 1-p])

    # Randomly replace elements in the matrix with new values
    new_values = np.random.randint(0, n_clusters, size=(n_categorical_features, n_samples))

    for i in range(n_categorical_features):
      for j in range(n_samples):
        if mask[i][j] == True: 
          if X_cat[i][j] == new_values[i][j]:
            X_cat[i][j] = (new_values[i][j]+1) % n_clusters
          else:
            X_cat[i][j] = new_values[i][j]

    
    # Generate random names for categorical features
    cat_names = [f"cat_feat_{i}" for i in range(n_categorical_features)]

    # Combine the numerical and categorical features
    df = pd.DataFrame(np.concatenate([X_num, X_cat.T], axis=1), columns=[f"num_feat_{i}" for i in range(n_numerical_features)] + cat_names)
    cat_feat_cols = [col for col in df.columns if 'cat_feat' in col]
    df[cat_feat_cols] = df[cat_feat_cols].astype(str) 

    # Add the target label
    df['target'] = cluster_idx
  
    # Save the dataset
    if save:
      df.to_csv(filename, index=False)

    return df