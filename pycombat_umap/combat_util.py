# @fefossa
import plotly.express as px
import pycytominer
from umap import UMAP
from combat.pycombat import pycombat
import pandas as pd

def pycombat_generator(df, well_column = 'Metadata_Well', batch_column = 'Metadata_Plate'):
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
    list_feats = pycytominer.cyto_utils.infer_cp_features(df, metadata=False)
    list_metadata = pycytominer.cyto_utils.infer_cp_features(df, metadata=True)
    
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

def generate_x_y_umap(df, n_neighbors = 5, min_dist = 0.1, metric='euclidean'):
    """
    This function will generate an X and y from the inputed dataframe and fit_transform based on the parameters specified
    *df (DataFrame): df containing the features as columns and samples as rows;
    *n_neighbors (int): number of neighbors for UMAP to calculate on 
        - The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
    *min_dist (int): The min_dist parameter controls how tightly UMAP is allowed to pack points together. Minimum distance apart that points are allowed to be in the low dimensional representation. 
        - This means that low values of min_dist will result in clumpier embeddings. 
    *col (str): 
    """

    feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)

    umap_2d = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, init='random', random_state=0, metric=metric)
    umap_2d.fit(X)
    projections = umap_2d.transform(X)

    proj_df = pd.DataFrame(data=projections, columns=['0', '1'])

    umap_df = pd.concat([proj_df, y], axis='columns')

    return umap_df

def plot_umap(df_plot, color_col, hover_cols, split_df = False, split_column = None, np = None, discrete = False, size=False, size_col = None, umap_param=False, neighbor=None, mindist=None, compound_color=False, time_color=False):
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
    """
    label_legend = color_col

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
    else:
        df['colors_plot_col'] = df[color_col]
        color_sequence = px.colors.qualitative.Vivid

    if size:
        df['Metadata_size'] = df[size_col].astype(int)
        title_plot = 'Size defined by '+ size_col
        if not time_color:
            df.sort_values('Metadata_size', inplace=True)

        fig = px.scatter(
        df, x='0', y='1',
        color='colors_plot_col',
        hover_data=hover_cols,
        color_continuous_scale=px.colors.sequential.Bluered,
        color_discrete_sequence=color_sequence,
        size='Metadata_size'
        )
    else:
        fig = px.scatter(
        df, x='0', y='1',
        color='colors_plot_col',
        hover_data=hover_cols,
        color_continuous_scale=px.colors.sequential.Bluered,
        color_discrete_sequence=color_sequence
        )
        fig.update_traces(marker={'size': 8})

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

    fig.show("notebook")

    return

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