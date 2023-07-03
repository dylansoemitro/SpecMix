import pandas as pd
from benchmark_algorithms import calculate_score
import warnings
warnings.filterwarnings("ignore")

def real_experiments(methods, metrics, num_clusters, kernel, numerical_cols, categorical_cols, 
                     path, column_names = None, sep=',', header=None, drop = None,  
                     lambdas=[], knn=0, scaling = True, sigmas = [], 
                     random_state = 0, n_init = 10, verbose = 0):
    scores = {}
    avg_time_taken = {}
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

    for m in methods:
        if m == 'spectral':
            continue
        score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, m, metrics=metrics, numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                                               random_state = random_state, n_init = n_init, verbose = verbose)
        scores[m] = score
        avg_time_taken[m] = time_taken
    if 'spectral' in methods:
        for ker in kernel:
            for l in lambdas:
                score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols),
                                                        numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=ker, scaling = scaling,
                                                        knn=knn,  sigmas = sigmas, random_state = random_state, n_init = n_init, verbose = verbose)
                scores[f'spectral lambda={l} kernel={ker}'] = score
                avg_time_taken[f'spectral lambda={l} kernel={ker}'] = time_taken

    scores_df = pd.DataFrame(scores)
    #first column is metrics
    scores_df.insert(0, 'metrics', metrics)
    avg_time_taken_df = pd.DataFrame(avg_time_taken, index=[0])
    return scores_df, avg_time_taken_df