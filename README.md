# Bio-image and data analysis scripts

The scripts and notebooks in this repository were created by @fefossa to support the projects developed during her Ph.D. It was developed during an internship at Cimini Lab and Carpenter-Singh Lab. 

Inside each folder, it contains a set of Python functions related to each subproject that can be applied to different use cases.

## Install

Clone this GitHub repository

```
git clone https://github.com/broadinstitute/scripts_notebooks_fossa.git
```

Then install using it:

```
pip install -e .
```

## Use

Inside each folder, there is an example notebook and an overall description.

To use any function inside a notebook, paste the following and run the cell:

```
%cd G:\My Drive\GitHub\scripts_notebooks_fossa
%pip install -e .
```

To import a utilitary file from any folder, for example:

```
from pycombat_umap import combat_util
```

## Details for each folder

### 1. Profile generator for CellProfiler and DeepProfiler outputs
**profiles folder:**
It has one folder for each software output, but the idea is the same for both. There are two notebooks:

- 1_Samples_retrieval.ipynb: get the single cells extracted from a database file (.sqlite) from all the plates in the batch, and join them into one CSV file;

- 2_AggAnnNormFeat.ipynb: from the single cell data, aggregate, annotate, normalize, and feature select the dataset using [pycytominer](https://github.com/cytomining/pycytominer). More details inside the notebook. 

### 2. Batch correct and visualize profiles
**pycombat_umap folder:**
It will process well-aggregated profiles and apply batch correction using PyCombat, and then use UMAP for visualization.
  
- **combat_util.py** file: functions that accept DataFrames (pandas library). The requirements are pycytominer, pandas, plotly.express, and UMAP.

- For more details on environment settings, see the readme inside the folder.

- Example of a plot:
<img src="https://github.com/broadinstitute/scripts_notebooks_fossa/assets/48028636/9b733dec-2939-4e2c-914d-0e8e8bd06021" width=50% height=50%>



### 3. Visualize samples replicability (mean average precision (mAP) results)
**plot_map folder:**
Give the main folder as an input, and looks in the subdirectories to find the files with the mAP x q values. 

- To calculate the mAP, use the instructions contained in the [evalzoo](https://github.com/cytomining/evalzoo/tree/main/matric).
  
- Then, use **plot_qvalue_map.ipynb** to plot the mAPs. Choose the title of the plot and save it.

- Example of the output:
<img src="https://github.com/broadinstitute/scripts_notebooks_fossa/assets/48028636/af9e866a-0706-46a8-80b2-38a7d170ed8c" width=50% height=50%>

### 4. Correlation matrix 
**correlation_matrix folder:** 
Here we have functions to calculate and generate a Pearson correlation matrix per plate or per dataset. 

<img src="https://github.com/broadinstitute/scripts_notebooks_fossa/assets/48028636/c7564db3-deb3-492c-b643-65196e0f017a" width=50% height=50%>


### 5. Dose-response (IC50)
**dose_response folder:** 
Create a dose-response curve based on concentration and cell viability values. Using linear regression, we calculate the linear function that represents that curve and get the IC50 (Inhibitory Concentration of 50% of the population). 

<img src="https://github.com/broadinstitute/scripts_notebooks_fossa/assets/48028636/bfb49eee-7c3c-4f8f-8ac4-1643709adfdd" width=50% height=50%>


