# Metadata contents

## 1. Using the apps: `app folder` 

## 1.1 Layout to CSV.exe: Transform a platemap into a metadata file

- ***IMPORTANT***: This app considers that you have images from a **Cytation 5** equipment (specific regex for that). 
- This app has the same code as `Layout_to_CSV.ipynb` notebook, just in a easy format to be used in a Windows machine. 

***INSTRUCTIONS***

- Download the app and double click. The program will execute;
    - Use the example below to understand how it works. 
    - INPUT: is a xlsx file (from Excel) containing a platemap just like the one you find [here](https://github.com/broadinstitute/scripts_notebooks_fossa/blob/main/metadata/app/Layout%20to%20CSV_example/layout_day1.xlsx). The treatments should be in the same position they would be in the actual plate.
    - INPUT: it identifies the first sequence of information and asks you what the column names will be based on the example;
    - OUTPUT: CSV and txt file just like the ones you find [here](https://github.com/broadinstitute/scripts_notebooks_fossa/tree/main/metadata/app/Layout%20to%20CSV_example/platemap).


***EXAMPLE***

1. Run the app and give `96` as the number of wells;
2. Choose the layout_day1.xlsx file inside `Layout to CSV_example` folder;
3. Answer what are the column's names based on the sequence of inputs inside one of the wells:
    - Huh7: cell
    - Amiodarone: compound
    - 1: concentration_uM
    - trt: control_type
4. Choose a folder to save the outputs;
5. It saves two files, one CSV and the other txt, with the information of what is contained within each well in the plate. 


## 1.2 Load Data Generator.exe. Transform a folder with images to a input file 

- ***IMPORTANT***: This app considers that you have images from a **Cytation 5** equipment (specific regex for that).
- This app has the same code as `Load_Data_Generator.ipynb` notebook, just in a easy format to be used in a Windows machine.

***INSTRUCTIONS***

- Download the app and double click. The program will execute:
        - Enter the number of wells in your plate;
        - INPUT: Give a folder containing images as an input;
        - Identifies the channels in that folder (DAPI, GFP, CY5, etc): answer which nomenclature you want for each channel;
        - OUTPUT: 3 CSV files with informations from that folder and metadata info.

***EXAMPLE***

- Find the outputs you'll get from using this app inside the `Load Data Generator_example` folder [here](https://github.com/broadinstitute/scripts_notebooks_fossa/tree/main/metadata/app/Load%20Data%20Generator_example).
    - `load_data.csv`: contains the FileName_Channel, PathName_Channel and the Metadata basic info (Plate, Well, Site);
    - `load_data_with_illum.csv` contains the same as above with the addition of PathName and FileName of the Illumination correction files (if using illumination correction pipeline);
    - `metadata_representative_cells.csv` to be used within the Basic protocol 2 available [here](https://currentprotocols.onlinelibrary.wiley.com/doi/full/10.1002/cpz1.713). 


## 2. Don't have a Windows machine to execute the app? 

- Use the notebooks `Load_Data_Generator.ipynb` or `Layout_to_CSV.ipynb`.

### 2.1 Requirements 

- Do the following on your environment:

 ```pip install easygui```