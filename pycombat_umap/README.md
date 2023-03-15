# PyCombat generator and functions to generate UMAP df and visualization

- These functions will process dataframes in order to:

1. Correct for batch effects using PyCombat [1]. It will deal with a dataframe that has features in the columns, and samples in the rows. 
2. Calculate the two UMAP vectors, based on the given dataframe. We use pycytominer to there are some requirements:
    - All columns that you'd like to keep after processing (for example, labels for compound, concentration, etc), should have a prefix "Metadata_";
    - All columns with actual features should have the prefixes Nuclei_, Cells_ and Cytoplasm_.
3. Plot the UMAP using the output from generate_x_y_umap with plotly. A lot of customization was added and more details are described within the function. 


[1] Behdenna A, Haziza J, Azencot CA and Nordor A. (2020) pyComBat, a Python tool for batch effects correction in high-throughput molecular data using empirical Bayes methods. bioRxiv doi: 10.1101/2020.03.17.995431