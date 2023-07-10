import pandas as pd
from examples.benchmark_algorithms import calculate_score
import warnings
warnings.filterwarnings("ignore")

def real_experiments(methods, metrics, num_clusters, path, kernel = [], numerical_cols=[], categorical_cols=[], 
                     column_names = None, sep=',', header=None, drop = None,  
                     lambdas=[], knn=0, scaling = True, sigmas = [], 
                     random_state = 0, n_init = 10, verbose = 0):
    scores = {}
    times_taken = {}
    # Load the CSV into a DataFrame
    df = pd.read_csv(path, names=column_names, sep=sep, header=header)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop the specified columns
    if drop:
        df = df.drop(drop, axis=1)
    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    df['target'] = df['target'].astype('category').cat.codes
    num_samples = df.shape[0]
    if num_samples == 0:
        raise ValueError('Dataset is empty after dropping rows with NaN values')
    num_clusters_detected = max(df['target'].tolist()) + 1
    if num_clusters_detected < num_clusters:
        print(f'Warning: Number of clusters detected in dataset after dropping missing values is {num_clusters_detected}, which is less than the specified number of clusters {num_clusters}')
        num_clusters = num_clusters_detected
    
    for m in methods:
        if m == 'specmix':
            continue
        score, time_taken = calculate_score(df, df['target'].tolist(), num_clusters, m, metrics=metrics, numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                                               random_state = random_state, n_init = n_init, verbose = verbose)
        scores[m] = score
        times_taken[m] = time_taken
    if 'specmix' in methods:
        for ker in kernel:
            for l in lambdas:
                score, time_taken = calculate_score(df, df['target'].tolist(), num_clusters, 'specmix',  metrics=metrics, lambdas=[l] * len(categorical_cols),
                                                        numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=ker, scaling = scaling,
                                                        knn=knn,  sigmas = sigmas, random_state = random_state, n_init = n_init, verbose = verbose)
                scores[f'specmix lambda={l} kernel={ker}'] = score
                times_taken[f'specmix lambda={l} kernel={ker}'] = time_taken

    scores_df = pd.DataFrame(scores)
    # add the average time taken for each method 
    time_taken_df = pd.DataFrame(times_taken, index = ['time_taken'])
    scores_df = pd.concat([scores_df, time_taken_df])

    return scores_df