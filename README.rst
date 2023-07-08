# SpecMix

SpecMix is a Python package that has implementations of SpecMix and onlyCat, clustering algorithms that rely on spectral graph theory for mixed-type and categorical data.

Both SpecMix and onlyCat are based off of the clustering algorithms in :code:`scikit-learn`.

## Usage

.. code:: python

    from SpecMix.specmix import SpecMix
    from examples.synthetic_dataset_generation import generate_mixed_dataset

    #Generate a synthetic dataset with 2 numerical features, 2 categorical features, 3 clusters, 0.1 noise
    df = generate_mixed_dataset(n_samples=1000, n_numerical_features=3, n_categorical_features=2, n_clusters=3, p=0.3)

    #Initialize the SpecMix algorithm with 3 clusters
    specmix = SpecMix(n_clusters=3, random_state=0)

    #Fit the algorithm to the dataset
    specmix.fit(df)

    #Print the cluster labels
    print(specmix.labels_)
    
This example shows the use of SpecMix on a synthetic dataset with 2 numerical features, 2 categorical features, 3 clusters, and 0.1 noise. The algorithm is initialized with 3 clusters and then fit to the dataset. The cluster labels are then printed. For more detailed usage instructions and examples, please refer to the demo.ipynb notebook included in this repository.

## Bugs and Issues

If you encounter any bugs or issues, feel free to `open an issue on the GitHub repository <https://github.com/dylansoemitro/SpecMix/issues>`_. Please include a detailed description of the bug or issue, as well as any relevant code snippets or error messages.


