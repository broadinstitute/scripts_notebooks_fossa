### PCA
import pandas as pd
import pycytominer
import plotly.express as px

def generate_pca(df, n_components = 2, noncanonical_features=False):
    """
    This function will generate PCA vectors based on the given dataframe.
    *df (DataFrame): df containing the features as columns and samples as rows;
    *n_components: the number of components used to reduce dimensionality;
    *noncanonical_features: set to True if the features don't have Nuclei, Cells, and Cytoplasm as a prefix.
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

def plot_pca(df_plot, color_col, hover_cols, split_df = False, split_column = None, 
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
    if df[color_col].map(type).eq(int).any():
        df.sort_values(color_col, inplace=True)

    if discrete:
        df['colors_plot_col'] = df[color_col].astype(str)
        color_sequence = px.colors.sequential.Plasma
        if time_color:
            color_sequence = ['royalblue', 'green','orange','red']
    elif dili_color:
        df['colors_plot_col'] = df[color_col]
        color_discrete = {"Aspirin": "#1f78b4", 'Amiodarone': "#33a02c", "Cyclophosphamide": "#e31a1c", "Etoposide": "#ff7f00",
                  "Vehicle-ETP":'rgb(204, 97, 176)', "Non-treated":'rgb(36, 121, 108)', "Lovastatin":"#6a3d9a", "Orphenadrine":"#a6cee3",
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

### EXAMPLE USE CASE

# import pca_utils
# plot_pca = pca_utils.generate_pca(df, n_components = 5)

# pca_utils.plot_pca(plot_pca, color_col='Metadata_compound', 
#                       hover_cols=["Metadata_Plate", "Metadata_Well"],
#                       size=True, size_col = "Metadata_concentration_uM",
#                       x="0", y="1"
#                       )