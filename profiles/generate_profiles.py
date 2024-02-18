"""
Functions to process profiles from single-cells or well-aggregated information

@fefossa
"""

import pycytominer
import pandas as pd
import pycytominer
import sqlite3
import easygui as eg
import os

def stringToBool(correlation_input):
    if correlation_input == 'yes':
       return True
    elif correlation_input == 'no':
       return False
    else:
      raise ValueError
    
def add_prefix_compartments(compartments, path, plate, list_cols):
    """
    Add prefixes to column names based on the compartments and concatenate the dataframes.

    Parameters:
    - compartments (list): List of compartment names.
    - path (str): Path to the folder containing compartment-specific CSV files.
    - plate (str): Plate identifier.
    - list_cols (list): List of column names to be excluded from the final dataframe.

    Returns:
    - pd.DataFrame: Concatenated dataframe with updated column names.
    """

    # Initialize an empty list to store dataframes for each compartment
    df_all = []

    # Iterate through each compartment
    for cmpt in compartments:
        # Read the CSV file for the current compartment
        df = pd.read_csv(rf"{path}\{plate}\{cmpt}.csv", low_memory=False)

        # Get list of metadata features and separate them from the main dataframe
        list_metadata = pycytominer.cyto_utils.infer_cp_features(df, metadata=True)
        meta = [df.pop(col) for col in list_metadata]
        df_meta = pd.concat(meta, axis=1)

        # Remove specified columns from the dataframe
        for name in list_cols:
            for cols in range(len(df.columns)):
                if name in df.columns[cols]:
                    df.drop(df.columns[cols], axis='columns', inplace=True)

        # Update column names by adding the compartment prefix
        for cols in range(len(df.columns)):
            df.rename(columns={df.columns[cols]: cmpt+"_"+df.columns[cols]}, inplace=True, errors='raise')

        # Concatenate metadata and updated dataframe
        df_after = pd.concat([df_meta, df], axis='columns')

        # Append the dataframe to the list
        df_all.append(df_after)

    # Concatenate all dataframes horizontally
    df_join = pd.concat(df_all, axis='columns')

    # Insert plate metadata at the beginning of the dataframe
    df_join.insert(0, 'Metadata_Plate', '')
    df_join['Metadata_Plate'] = plate

    return df_join

def merge_sqlite_dataframes(plate_list, common_path, compartments):
    """
    Merge data from SQLite database tables for multiple plates and compartments.

    Parameters:
    - plate_list (list): List of plate identifiers.
    - common_path (str): Common path to the folder containing SQLite databases.
    - compartments (list): List of compartment names.

    Returns:
    - list of pd.DataFrame: List of dataframes containing merged data for each plate.
    """

    # Initialize an empty list to store the merged dataframes
    df_join = []

    # Iterate through each plate
    for plate in plate_list:
        # Connect to the SQLite database for the current plate
        conn = sqlite3.connect(fr"{common_path}/{plate}/{plate}.db") # < ------------ mudar padrão se for diferente o caminho pro arquivo
        conn_cursor = conn.cursor()

        # Read metadata information from the 'Per_Image' table
        df_image = pd.read_sql_query("SELECT * FROM Per_Image", conn)
        df_image.rename(columns={'Image_Metadata_Site': 'Metadata_Site',
                         'Image_Metadata_Plate': 'Metadata_Plate',
                         'Image_Metadata_Well': 'Metadata_Well'}, inplace=True)
        print('Metadata from image df: DONE')

        # Extract relevant metadata columns
        metadata_cols = df_image[['ImageNumber', 'Metadata_Plate', 'Metadata_Well', 'Metadata_Site']]

        # Initialize an empty list to store dataframes for each compartment
        df_list = []

        # Iterate through each compartment
        for eachcompartment in compartments:
            # Read data from the compartment-specific table
            df_temp = pd.read_sql_query(f"SELECT * FROM Per_{eachcompartment}", conn)
            print(f'df for compartment {eachcompartment}: DONE')

            # Append the compartment dataframe to the list
            df_list.append(df_temp)

        # Concatenate dataframes for each compartment horizontally
        df = pd.concat(df_list, axis='columns')

        # Remove duplicated columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Merge metadata columns with the compartment dataframe based on 'ImageNumber'
        df = metadata_cols.merge(df, on='ImageNumber', how="right")

        # Append the merged dataframe to the final list
        df_join.append(df)

    return df_join

def profiling_inputs():
   """
   """
   profile = eg.fileopenbox(msg="Choose a file with samples and their features", default=r"F:")
   print('Filename', profile)

   project_name = eg.enterbox("Provide the name of this project:")
   print('Project name:', project_name)

   metadata_question = eg.enterbox("If you need to annotate your dataset with an external file, write yes and press enter. If already annotated, answer no and press enter.")
   metadata_answer = stringToBool(metadata_question)

   if metadata_answer:
      platemap = eg.fileopenbox(msg="Choose a map (csv file) with plates names and metadata filenames", default=r"F:")
      platemap_path = os.path.split(platemap)[0]
      print('Platemap file selected', platemap)
      barcode_df = pd.read_csv(platemap)
   else:
      barcode_df = None
      platemap_path = None

   return profile, project_name, barcode_df, platemap_path