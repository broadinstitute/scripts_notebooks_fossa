# Samples retrieval from a database file and generate profiles based on the single-cells file

- These notebooks will process dataframes in order to:

aggregate => 1_Samples_retrieval.ipynb: will take a database file as an input and extract and merge the single_cells from different tables inside the .db file. Export an csv file with single-cell information.

aggregate => 2_AggAnnNormFeat.ipynb: will use the single-cells file obtained in the first notebook, aggregate, annotate, normalize and feature select using pycytominer. 

- Requirements:

pycytominer
easygui
