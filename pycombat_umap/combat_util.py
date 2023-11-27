# @fefossa
import plotly.express as px
import pycytominer
from umap import UMAP
from combat.pycombat import pycombat
import pandas as pd

def pycombat_generator(df, well_column = 'Metadata_Well', batch_column = 'Metadata_Plate', noncanonical_features=False):
    """
    This function will take the DataFrame containing features (columns) and samples (rows), transpose it to have the samples as the columns with a common denominator.
    For example, if the Metadata_Plate is the batch column, then the name of the tranposed columns will be the Metadata_Well + Metadata_Plate. 
    
    Inputs
    *df (DataFrame): df with features as columns and samples as rows.
    *well_column (str): the name of the column where the Well information is contained.
    *batch_column (str): the name of the column that determines the batch effect observed.
        - This will usually be different days of experiment, plates, etc. 

    """
    #generate transposed df summing everything
    df['Metadata_TranposedSamples'] = df[well_column].astype(str) + '_' + df[batch_column].astype(str)
    
    #get list of metadata and features columns
    list_metadata = pycytominer.cyto_utils.infer_cp_features(df, metadata=True)
    if noncanonical_features:
        list_feats = [x for x in df.columns.tolist() if x not in list_metadata]
    else:
        list_feats = pycytominer.cyto_utils.infer_cp_features(df, metadata=False)
    
    
    #set a new index to the df based on Metadata_TranposedSamples and transpose it 
    df_T = df[list_feats].reset_index(drop=True).set_index(df['Metadata_TranposedSamples']).T
    
    #takes the column which represents the batch and create a batch list from it
    #pycombat needs this to know what you want to factor out
    batch_col_list = list(df[batch_column].unique())

    batch = []
    for pl in range(len(batch_col_list)):
        df_plate = df.loc[df[batch_column] == batch_col_list[pl]].reset_index(drop=True)
        for j in range(len(df_plate.index)):
            batch.append(pl)

    #calculate the corrected dataframe 
    df_corrected = pycombat(df_T, batch)

    #transpose the table back and join with metadata dataframe
    df_corr_T = df_corrected.T.reset_index(drop=True)
    df_metadata = df[list_metadata]
    df_corr = pd.concat([df_metadata, df_corr_T], axis='columns', join='inner')
    
    return df_corr

def generate_x_y_umap(df, n_neighbors = 5, min_dist = 0.1, n_components = 2, metric='cosine', noncanonical_features=False, iterate=False, number_runs=None):
    """
    This function will generate an X and y from the inputed dataframe and fit_transform based on the parameters specified
    *df (DataFrame): df containing the features as columns and samples as rows;
    *n_neighbors (int): number of neighbors for UMAP to calculate on 
        - The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
    *min_dist (int): The min_dist parameter controls how tightly UMAP is allowed to pack points together. Minimum distance apart that points are allowed to be in the low dimensional representation. 
        - This means that low values of min_dist will result in clumpier embeddings. 
    *col (str): 
    """
    import numpy as np
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    if noncanonical_features:
        feat = [x for x in df.columns.tolist() if x not in meta]
    else:
        feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
        
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)

    umap_2d = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, init='random', random_state=0, metric=metric)
    umap_2d.fit(X)
    projections = umap_2d.transform(X)
    columns = [str(x) for x in range(0, n_components)]

    if iterate:
        # Create an empty list to store UMAP embeddings
        umap_embeddings = []

        # Number of UMAP runs
        num_runs = number_runs

        for _ in range(num_runs):
            umap_2d = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, init='random', random_state=0, metric=metric)
            embedding_umap = umap_2d.fit_transform(X)  # Replace your_data with your actual data
            umap_embeddings.append(embedding_umap)
        
        # Convert the list to a NumPy array for easier manipulation
        umap_results_array = np.array(umap_embeddings)

        # Calculate mean and standard deviation along each dimension
        mean_embedding = np.mean(umap_results_array, axis=0)
        std_embedding = np.std(umap_results_array, axis=0)
        
        # Dataframes
        error_df = pd.DataFrame(data=std_embedding, columns=['x_err', 'y_err'])
        proj_df = pd.DataFrame(data=mean_embedding, columns=columns)
        umap_df = pd.concat([proj_df, error_df, y], axis='columns')
    else:
        proj_df = pd.DataFrame(data=projections, columns=columns)
        umap_df = pd.concat([proj_df, y], axis='columns')

    return umap_df

def plot_umap(df_plot, color_col, hover_cols, split_df = False, split_column = None, np = None, discrete = False, size=False, size_col = None, umap_param=False, neighbor=None, mindist=None, compound_color=False, time_color=False, dili_color=False,
              x="0", y="1", symbol="symbol", error_x=None, error_y=None):
    """
    Plot UMAP components to visualize results using plotly scatter function. Various parameters are optional to choose and customize the plots.
    
    Inputs
    *df_plot (DataFrame): df that contains labels for each row and the 0 and 1 vector for UMAP plot. 
    *color_col (str): column that will give the color coding for the plot.
    *hover_cols (list): which columns will show information when hovering over the points.
    
    Optional
    *split_df (bool): True if want to filter the df based on a column
    *split_column (str): name of the column you'd like the split to occur
    *np (str): the name of the sample you'd like to plot that's inside the split_column
    *discrete (bool): True to plot a discrete distribution with a continuous color sequence 
        - In cases where you'd have a range of concentrations and you'd like each one to show as individual points,
        but with a gradient color scheme.     
    *umap_param (bool): if True, it will change the title to have the number of neighbors and min_dist parameters from umap displayed.
        *neighbor (int): number used as n_neighbors
        *mindist (int): number used as min_dist
    *size (bool): if True, use the values of a column to determine the size of the scatter points.
        size_col (str): provide the name of the column being used as the size denominator.
    *compound_color (bool): 
    *time_color (bool): if plotting Cell Recovery data, use this colormap to the 4 timepoints available in this dataset. 
    *dili_color (bool): if plotting DILI experiments results, use the color sequence previously defined on a dictionary. Used to maintain the same pattern across plots.
    """
    label_legend = color_col
    color_discrete={}
    if split_df:
        df = df_plot[df_plot[split_column] == np].reset_index()
        title_plot = np
        if umap_param:
            title_plot = np + ' N: ' + str(neighbor) + ' M: ' + str(mindist)
    else:
        df = df_plot.copy()
        title_plot = 'Labeled by '+ color_col
        if umap_param:
            title_plot = 'N: ' + str(neighbor) + ' M: ' + str(mindist)
    #if the color_col is a int, sort the columns by value
    if df[color_col].map(type).eq(int).any():
        df.sort_values(color_col, inplace=True)

    if discrete:
        df['colors_plot_col'] = df[color_col].astype(str)
        color_sequence = px.colors.sequential.Plasma
        if compound_color:
            color_sequence = px.colors.sequential.Hot
        if time_color:
            color_sequence = ['royalblue', 'green','orange','red']
            label_legend = 'Time of cell recovery<br>after AgNP treatment<br>(in days)'
    elif dili_color:
        df['colors_plot_col'] = df[color_col]
        color_discrete = {"Aspirin": 'rgb(229, 134, 6)', 'Amiodarone': 'rgb(93, 105, 177)', "Cyclophosphamide": 'rgb(82, 188, 163)', "Etoposide": 'rgb(153, 201, 69)',
                  "Vehicle-ETP":'rgb(204, 97, 176)', "Non-treated":'rgb(36, 121, 108)', "Lovastatin":'rgb(218, 165, 27)', "Orphenadrine":'rgb(47, 138, 196)',
                  "Tetracycline":'rgb(118, 78, 159)', "DMSO":'rgb(237, 100, 90)', "Lactose":'rgb(165, 170, 153)'}
        color_sequence = px.colors.qualitative.Vivid
    else:
        df['colors_plot_col'] = df[color_col]
        color_sequence = px.colors.qualitative.Vivid

    if size:
        df['Metadata_size'] = df[size_col].astype(int)
        title_plot = 'Size defined by '+ size_col
        if not time_color:
            df.sort_values('Metadata_size', inplace=True)
        if dili_color:
            fig = px.scatter(
            df, x=x, y=y,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_map=color_discrete,
            error_x=error_x, error_y=error_y
            )
        else:
            fig = px.scatter(
            df, x=x, y=y,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_sequence=color_sequence
            )
    else:
        fig = px.scatter(
        df, x=x, y=y,
        color='colors_plot_col',
        hover_data=hover_cols,
        color_continuous_scale=px.colors.sequential.Bluered,
        color_discrete_sequence=color_sequence,
        error_x=error_x, error_y=error_y
        )
        fig.update_traces(marker={'size': 5})

    fig.update_layout(
        dict(updatemenus=[
                        dict(
                            type = "buttons",
                            direction = "left",
                            buttons=list([
                                dict(
                                    args=["visible", "legendonly"],
                                    label="Deselect All",
                                    method="restyle"
                                ),
                                dict(
                                    args=["visible", True],
                                    label="Select All",
                                    method="restyle"
                                )
                            ]),
                            pad={"r": 5, "t": 5},
                            showactive=True,
                            x=1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        ),
                    ]
              ),
        font=dict(
        size=18),
        legend_title=label_legend,
        title=title_plot,
        autosize=False,
        width=900,
        height=700,
        margin=dict(
            l=50,
            r=50,
            b=20,
            t=50,
            pad=4
        )
        )
    fig.update_traces(error_x=dict(thickness=3), error_y=dict(thickness=3))
    # fig.show("notebook")

    return fig.show("notebook")

def plot_umap_3d(df_plot, color_col, hover_cols, split_df = False, split_column = None, np = None, discrete = False, size=False, size_col = None, umap_param=False, neighbor=None, mindist=None, compound_color=False, time_color=False, dili_color=False,
              x="0", y="1", z="2"):
    """
    Plot UMAP components to visualize results using plotly scatter function. Various parameters are optional to choose and customize the plots.
    
    Inputs
    *df_plot (DataFrame): df that contains labels for each row and the 0 and 1 vector for UMAP plot. 
    *color_col (str): column that will give the color coding for the plot.
    *hover_cols (list): which columns will show information when hovering over the points.
    
    Optional
    *split_df (bool): True if want to filter the df based on a column
    *split_column (str): name of the column you'd like the split to occur
    *np (str): the name of the sample you'd like to plot that's inside the split_column
    *discrete (bool): True to plot a discrete distribution with a continuous color sequence 
        - In cases where you'd have a range of concentrations and you'd like each one to show as individual points,
        but with a gradient color scheme.     
    *umap_param (bool): if True, it will change the title to have the number of neighbors and min_dist parameters from umap displayed.
        *neighbor (int): number used as n_neighbors
        *mindist (int): number used as min_dist
    *size (bool): if True, use the values of a column to determine the size of the scatter points.
        size_col (str): provide the name of the column being used as the size denominator.
    *compound_color (bool): 
    *time_color (bool): if plotting Cell Recovery data, use this colormap to the 4 timepoints available in this dataset. 
    *dili_color (bool): if plotting DILI experiments results, use the color sequence previously defined on a dictionary. Used to maintain the same pattern across plots.
    """
    label_legend = color_col
    color_discrete={}
    if split_df:
        df = df_plot[df_plot[split_column] == np].reset_index()
        title_plot = np
        if umap_param:
            title_plot = np + ' N: ' + str(neighbor) + ' M: ' + str(mindist)
    else:
        df = df_plot.copy()
        title_plot = 'Labeled by '+ color_col
        if umap_param:
            title_plot = 'N: ' + str(neighbor) + ' M: ' + str(mindist)
    #if the color_col is a int, sort the columns by value
    if df[color_col].map(type).eq(int).any():
        df.sort_values(color_col, inplace=True)

    if discrete:
        df['colors_plot_col'] = df[color_col].astype(str)
        color_sequence = px.colors.sequential.Plasma
        if compound_color:
            color_sequence = px.colors.sequential.Hot
        if time_color:
            color_sequence = ['royalblue', 'green','orange','red']
            label_legend = 'Time of cell recovery<br>after AgNP treatment<br>(in days)'
    elif dili_color:
        df['colors_plot_col'] = df[color_col]
        color_discrete = {"Aspirin": 'rgb(229, 134, 6)', 'Amiodarone': 'rgb(93, 105, 177)', "Cyclophosphamide": 'rgb(82, 188, 163)', "Etoposide": 'rgb(153, 201, 69)',
                  "Vehicle-ETP":'rgb(204, 97, 176)', "Non-treated":'rgb(36, 121, 108)', "Lovastatin":'rgb(218, 165, 27)', "Orphenadrine":'rgb(47, 138, 196)',
                  "Tetracycline":'rgb(118, 78, 159)', "DMSO":'rgb(237, 100, 90)', "Lactose":'rgb(165, 170, 153)'}
        color_sequence = px.colors.qualitative.Vivid
    else:
        df['colors_plot_col'] = df[color_col]
        color_sequence = px.colors.qualitative.Vivid

    if size:
        df['Metadata_size'] = df[size_col].astype(int)
        title_plot = 'Size defined by '+ size_col
        if not time_color:
            df.sort_values('Metadata_size', inplace=True)
        if dili_color:
            fig = px.scatter_3d(
            df, x=x, y=y, z=z,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_map=color_discrete
            )
        else:
            fig = px.scatter_3d(
            df, x=x, y=y, z=z,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_sequence=color_sequence
            )
    else:
        fig = px.scatter_3d(
        df, x=x, y=y, z=z,
        color='colors_plot_col',
        hover_data=hover_cols,
        color_continuous_scale=px.colors.sequential.Bluered,
        color_discrete_sequence=color_sequence
        )
        fig.update_traces(marker={'size': 12})

    fig.update_layout(
        dict(updatemenus=[
                        dict(
                            type = "buttons",
                            direction = "left",
                            buttons=list([
                                dict(
                                    args=["visible", "legendonly"],
                                    label="Deselect All",
                                    method="restyle"
                                ),
                                dict(
                                    args=["visible", True],
                                    label="Select All",
                                    method="restyle"
                                )
                            ]),
                            pad={"r": 5, "t": 5},
                            showactive=True,
                            x=1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        ),
                    ]
              ),
        font=dict(
        size=18),
        legend_title=label_legend,
        title=title_plot,
        autosize=False,
        width=900,
        height=700,
        margin=dict(
            l=50,
            r=50,
            b=20,
            t=50,
            pad=4
        )
        )

    # fig.show("notebook")

    return fig.show("notebook")
 
def umap_search(df, n_neighbors_list = [5, 15, 30, 50], min_dist_list = [0, 0.01, 0.05, 0.1, 0.5, 1]):
    """
    """
    df_temp=[]
    for n in n_neighbors_list:
        for m in min_dist_list:
            df_plot = generate_x_y_umap(df, n_neighbors=n, min_dist=m)
            df_temp.append(df_plot)

    return df_temp

def seaborn_plot(df, color_col ='Metadata_Plate'):
    """
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # x_label="pca_2"
    # y_label="pca_3"
    font_size=40
    sns.set(rc={'figure.figsize':(20, 15)})
    b = sns.scatterplot(data=df, x='0', y='1', hue=color_col, s=200)
    # b.axes.set_title("Title",fontsize=50)
    b.set_xlabel(fontsize=font_size)
    b.set_ylabel(fontsize=font_size)
    b.tick_params(labelsize=font_size)
    plt.legend(loc="upper right", frameon=True, fontsize=font_size)
    plt.show()
    return

def generate_batch_col(df, rows=False,
                       columns=False, 
                       change_default_dict = False,
                       dict_change = None,
                       join_plate=False):
    """
    Based on the Metadata_Well column, get the number of the columns in the plate (2, 3, 4...) and generate a batch_column containing the index that represents each batch (1, 2, 3...).
    Provide a dictionary containing the batch you'd like to consider from the dictionary.
    *dict_columns (default dict): {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, '11':9}
    *dict_rows (default dict): {'B':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5}
    """
    import re
    well_list = df['Metadata_Well'].tolist()
    get_values = []
    for wells in well_list:
        splited = re.split('(\d+)',wells)
        if columns:
            get_values.append(splited[1])
        if rows:
            get_values.append(splited[0])
    df['Metadata_values'] = get_values

    if not change_default_dict:
        if rows:
            dict_change = {'B':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5}
        if columns:
            dict_change = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, '11':9}

    df['Metadata_batch_col'] = df['Metadata_values']
    df.replace({'Metadata_batch_col':dict_change}, inplace=True)

    if join_plate:
        df['Metadata_batch_pycombat'] = df['Metadata_batch_col'].astype(str) + '_' + df['Metadata_Plate']
    else:
        df['Metadata_batch_pycombat'] = df['Metadata_batch_col']
    
    # print(df['Metadata_batch_pycombat'])

    return df

def col_generator(df, cols_to_join = ['Metadata_Compound', 'Metadata_Concentration']):
    """
    Create a new column containing information from compound + concentration of compounds
    *cols_to_join: provide columns names to join on, order will be determined by order in this list
    """
    col_copy = cols_to_join.copy()
    init = cols_to_join.pop(0) #pop the first element of the list
    new_col_temp = [init] #keep the first element in the list
    for cols in cols_to_join:
        temp = cols.split("_", 1) #only split metadata out
        print(temp[1])
        new_col_temp.append(temp[1])
    new_col = ('_'.join(new_col_temp))  #generate the new column name from the list
    df[new_col] = df[col_copy].astype(str).agg(' '.join, axis=1) #transform the column to str and create new metadata
    print("Names of the compounds + concentration: ",  df[new_col].unique())

    return df, new_col

def tsne_generator(df, perplexity=40, n_components = 2, metric='cosine', noncanonical_features=False, iterate=False, number_runs=None):
    """ 
    
    """
    from sklearn.manifold import TSNE
    import numpy as np

    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    if noncanonical_features:
        feat = [x for x in df.columns.tolist() if x not in meta]
    else:
        feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
        
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    print(tsne.kl_divergence_)

    columns = [str(x) for x in range(0, n_components)]

    if iterate:
        # Create an empty list to store UMAP embeddings
        tsne_embeddings = []

        # Number of UMAP runs
        num_runs = number_runs

        for _ in range(num_runs):
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
            embedding_tsne = tsne.fit_transform(X)  # Replace your_data with your actual data
            print(tsne.kl_divergence_)
            tsne_embeddings.append(embedding_tsne)
        
        # Convert the list to a NumPy array for easier manipulation
        tsne_results_array = np.array(tsne_embeddings)

        # Calculate mean and standard deviation along each dimension
        mean_embedding = np.mean(tsne_results_array, axis=0)
        std_embedding = np.std(tsne_results_array, axis=0)
        
        # Dataframes
        error_df = pd.DataFrame(data=std_embedding, columns=['x_err', 'y_err'])
        proj_df = pd.DataFrame(data=mean_embedding, columns=columns)
        tsne_df = pd.concat([proj_df, error_df, y], axis='columns')
    else:
        proj_df = pd.DataFrame(data=X_tsne, columns=columns)
        tsne_df = pd.concat([proj_df, y], axis='columns')

    return X, tsne_df

def tsne_divergence(X_train, range=80):

    import numpy as np
    from sklearn.manifold import TSNE

    perplexity = np.arange(5, range, 5)
    divergence = []

    for i in perplexity:
        model = TSNE(n_components=2, init="pca", perplexity=i, random_state=42)
        reduced = model.fit_transform(X_train)
        divergence.append(model.kl_divergence_)
    fig = px.line(x=perplexity, y=divergence, markers=True)
    fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
    fig.update_traces(line_color="red", line_width=1)
    fig.show()

def plot_tsne(df_plot, color_col, hover_cols, split_df = False, split_column = None, 
              np = None, discrete = False, size=False, size_col = None, umap_param=False, 
              neighbor=None, mindist=None, time_color=False,
              x="0", y="1", label_legend = "", title_plot="",
              symbol=False, symbol_col=None, symbol_list=None,
              error_x=None, error_y=None, dili_color=False):

    if split_df:
        df = df_plot[df_plot[split_column] == np].reset_index()
        title_plot = np
        if umap_param:
            title_plot = np + ' N: ' + str(neighbor) + ' M: ' + str(mindist)
    else:
        df = df_plot.copy()
        # title_plot = 'Labeled by '+ color_col
        # if umap_param:
        #     title_plot = 'N: ' + str(neighbor) + ' M: ' + str(mindist)
    #if the color_col is a int, sort the columns by value
    if df[color_col].map(type).eq(int).any():
        df.sort_values(color_col, inplace=True)

    if discrete:
        df['colors_plot_col'] = df[color_col].astype(str)
        color_sequence = px.colors.sequential.Plasma
        if time_color:
            color_sequence = ['royalblue', 'green','orange','red']
    elif dili_color:
        df['colors_plot_col'] = df[color_col]
        color_discrete = {"Aspirin": 'rgb(229, 134, 6)', 'Amiodarone': 'rgb(93, 105, 177)', "Cyclophosphamide": 'rgb(82, 188, 163)', "Etoposide": 'rgb(153, 201, 69)',
                  "Vehicle-ETP":'rgb(204, 97, 176)', "Non-treated":'rgb(36, 121, 108)', "Lovastatin":'rgb(218, 165, 27)', "Orphenadrine":'rgb(47, 138, 196)',
                  "Tetracycline":'rgb(118, 78, 159)', "DMSO":'rgb(237, 100, 90)', "Lactose":'rgb(165, 170, 153)'}
        color_sequence = px.colors.qualitative.Vivid
    else:
        df['colors_plot_col'] = df[color_col]
        color_sequence = px.colors.qualitative.Vivid

    if size:
        df['Metadata_size'] = df[size_col].astype(int)
        if not time_color:
            df.sort_values('Metadata_size', inplace=True)
        if symbol:
            fig = px.scatter(
            df, x=x, y=y,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_sequence=color_sequence,
            symbol=symbol_col,
            symbol_sequence=symbol_list
            )
        if dili_color:
            fig = px.scatter(
            df, x=x, y=y,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_map=color_discrete,
            error_x=error_x, error_y=error_y
            )
            fig.update_traces(marker=dict(
                              line=dict(width=1,
                                        color='White')),
                  selector=dict(mode='markers'))

        else:
            fig = px.scatter(
            df, x=x, y=y,
            color='colors_plot_col',
            hover_data=hover_cols,
            color_continuous_scale=px.colors.sequential.Bluered,
            size='Metadata_size',
            color_discrete_sequence=color_sequence,
            error_x=error_x, error_y=error_y
            )
    else:
        fig = px.scatter(
        df, x=x, y=y,
        color='colors_plot_col',
        hover_data=hover_cols,
        color_continuous_scale=px.colors.sequential.Bluered,
        color_discrete_sequence=color_sequence,
        error_x=error_x, error_y=error_y
        )
        fig.update_traces(marker=dict(
                                      size=7,
                              line=dict(width=1,
                                        color='White')),
                  selector=dict(mode='markers'))

    fig.update_layout(plot_bgcolor='white',
        font=dict(
        size=18),
        legend_title=label_legend,
        title=title_plot,
        autosize=False,
        width=900,
        height=700,
        margin=dict(
            l=50,
            r=50,
            b=20,
            t=50,
            pad=4
        ),
        xaxis_title="TSNE X",
        yaxis_title="TSNE Y"
        )
    
    fig.update_traces(error_x=dict(thickness=2, color='black'), error_y=dict(
        color='black',
        thickness=2,
    ),)
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
    )

    # fig.show("notebook")
    config = {
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 500,
        'width': 700,
        'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
    }
    }

    return fig.show(config=config)

def generate_pca(df, n_components = 2, noncanonical_features=False):
    """
    This function will generate an X and y from the inputed dataframe and fit_transform based on the parameters specified
    *df (DataFrame): df containing the features as columns and samples as rows;
    *n_neighbors (int): number of neighbors for UMAP to calculate on 
        - The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
    *min_dist (int): The min_dist parameter controls how tightly UMAP is allowed to pack points together. Minimum distance apart that points are allowed to be in the low dimensional representation. 
        - This means that low values of min_dist will result in clumpier embeddings. 
    *col (str): 
    """
    from sklearn.decomposition import PCA

    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    if noncanonical_features:
        feat = [x for x in df.columns.tolist() if x not in meta]
    else:
        feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
        
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)

    pca = PCA(n_components=n_components)

    # Fit and transform the data
    pca_result = pca.fit_transform(X)

    # Access the principal components and explained variance ratio
    components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_

    # Print the results
    print(f"Explained Variance Ratio: {explained_variance_ratio}")

    columns = [str(x) for x in range(0, n_components)]

    proj_df = pd.DataFrame(data=pca_result, columns=columns)

    pca_df = pd.concat([proj_df, y], axis='columns')

    return pca_df