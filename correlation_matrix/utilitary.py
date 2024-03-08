"""
These utilitary file contains functions to calculate correlation matrix, get Pearson coefficient values and plot them 
We can also plot dose response curves and calculate the EC50

@author fefossa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import curve_fit
import pycytominer

def log_ec50(ecf, b, f):
    """
    ecf as log(ecf): ECf is the concentration of agonist that gives a response F percent of the way between Bottom and Top
    b as hill slope
    F as the constrain (which EC you would like to predict)
    """
    return np.log(ecf) - ((1/b)*np.log(f/(100-f)))

def logistic_curve(x, b, c, d, logec50):
    """
    x is the dose value already converted to log(x)
    b: hill slope
    c: bottom value
    d: top value
    logec50: output from log_ec50 function
    returns y response value
    """
    return c + (d - c)/(1+10**((logec50-x)*b)) 

def corr_matrix_per_plate(df_input, plates, plot_joined_replicates=False, metadata_column = 'Cells_CompoundSizeCnc', filter = False, filter_col = None, filter_list = None):
    """
    Calculate a correlation matrix in a per-plate basis. Takes a list containing dataframes inside of it, calculates the corr matrix, and then plot each correlation matrix
    *df (dataframe) is the dataframe already normalized and feature selected
    *plates (list) is a list with the name of each plate you have in your assay/you want to calculate pearson coefficient to
    """
    cols_keep = pycytominer.cyto_utils.features.infer_cp_features(df_input, metadata=False)
    cols_keep.append(metadata_column)
    #split df by plates
    if filter:
        df = df_input[df_input[filter_col].isin(filter_list)].reset_index()
    else:
        df = df_input.copy()

    df_temp_list = []
    for pl in plates:
        df_temp = df.loc[df['Metadata_Plate'] == pl]
        df_temp_list.append(df_temp)
        print('Shape of each DataFrame, split by plate', df_temp.shape)
    corr = []
    if plot_joined_replicates:
        df_temp = pd.concat(df_temp_list)
        df_select_cols = df_temp[cols_keep]
        df_transposed = df_select_cols.set_index(metadata_column).T
        corr_temp = df_transposed.corr()
        corr.append(corr_temp)
    else:
        #select only the columns from the compartments + one categorical column (the Labels)
        df_select_cols = []
        for d in df_temp_list:
            df_TT = d[cols_keep]
            df_select_cols.append(df_TT)
        # print(df_select_cols)
        #Set index as the metadata_column we created before and transpose the df
        df_transposed = []
        for d in df_select_cols:
            df_T = d.set_index(metadata_column).T
            df_transposed.append(df_T)
        #3 calculate corr matrix per-plate basis
        corr = []
        for d in df_transposed:
            df_corr = d.corr()
            corr.append(df_corr)
    return corr

def plot_corr_matrix(corr, labelsize=7):
    """
    Plot the correlation matrix based on a list of dataframes
    *corr is a list containing dataframes, each df being a correlation matrix
    """
    #plot corr matrix per plate
    for d in corr:
        sorted_df = d.sort_index(ascending=True, axis = 'columns')
        df_corr = sorted_df.sort_index(ascending=True, axis = 'index')
        colormap = sns.color_palette("coolwarm", as_cmap=True)
        plt.figure(figsize = (35, 30))
        plt.tick_params(axis='both', which='major', labelsize=labelsize, labelbottom = False, bottom=False, top = False, labeltop=True)
        fig = sns.heatmap(df_corr, 
                xticklabels=df_corr.columns,
                yticklabels=df_corr.columns,
                cmap = colormap,
                annot=False,
                vmin=-1, vmax=1)
        fig.set_xlabel('')
        plt.ylabel('Compound_Concentration \n', fontsize=16)
    
    return

def plot_cluster_map(corr, labelsize=7):
    """
    Plot the correlation matrix based on a list of dataframes
    *corr is a list containing dataframes, each df being a correlation matrix
    """
    #plot corr matrix per plate
    for d in corr:
        sorted_df = d.sort_index(ascending=True, axis = 'columns')
        df_corr = sorted_df.sort_index(ascending=True, axis = 'index')
        colormap = sns.color_palette("coolwarm", as_cmap=True)
        # plt.tick_params(axis='both', which='major', labelsize=labelsize)
        fig = sns.clustermap(df_corr, 
                             figsize = (35, 30),
                xticklabels=df_corr.columns,
                yticklabels=df_corr.columns,
                cmap = colormap,
                annot=False,
                vmin=-1, vmax=1)
        plt.ylabel('Compound_Concentration \n', fontsize=16)

        return

def correlation_pairs(df_corr):
    """
    Generate the correlation pairs from the corr dataframes using the unstack functions, removing the NaN values. 
    Retain upper triangular values of correlation matrix and make Lower triangular values Null.
    *df_corr is the list of dataframes containing the corr matrices for each plate
    Return is the unstacked information for each dataframe inside df_corr
    """
    unstacked = []
    for corr in df_corr:
        upper_corr_mat = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)) #retain upper triangular values of correlation matrix and make Lower triangular values Null
        
        #Convert to 1-D series and drop Null values
        unique_corr_pairs = upper_corr_mat.unstack().dropna()
        
        # Sort correlation pairs
        sorted_mat = unique_corr_pairs.sort_values()
        unstacked.append(sorted_mat)
    
    return unstacked

def create_dict(df, list_cmp, dmso = False, negcon = False, poscon = False):
    """
    Create dictionaries for AgNP 40 and AgNP 100 treated cells. 
    You can choose to get Pearson coefficient values from replicates similarities or similarity to DMSO (positive control)
    *df: contains correlated values organized in a unstacked form (output from unstack() function from pandas)
    *dmso: set to True if you want to see similarity of treated cells to DMSO. Set to False to see similarities between replicates.
    *list_cmp is a list with the compounds names
    """ 
    main_dict = []
    for plate in range(len(df)):
        dict_40 = {}
        dict_100 = {}
        for cmp in list_cmp:
            keep = []
            keep_negcon = []
            keep_poscon = []
            if dmso:
                for val in range(len(df[plate][cmp].loc['DMSO 0.0 10.0'])):
                    keep.append(df[plate][cmp].loc['DMSO 0.0 10.0'][val])
            else:
                for val in range(len(df[plate][cmp].loc[cmp])):
                    keep.append(df[plate][cmp].loc[cmp][val])
                for val in range(len(df[plate]['Non-treated 0.0 0.0'].loc['Non-treated 0.0 0.0'])):
                    keep_negcon.append(df[plate]['Non-treated 0.0 0.0'].loc['Non-treated 0.0 0.0'][val])
                for val in range(len(df[plate]['DMSO 0.0 10.0'].loc['DMSO 0.0 10.0'])):
                    keep_poscon.append(df[plate]['DMSO 0.0 10.0'].loc['DMSO 0.0 10.0'][val])
            if 'AgNP 40' in cmp:
                dict_40[cmp] = keep
                if negcon:
                    dict_40['Non-treated 0 0.0'] = keep_negcon
                if poscon:
                    keep_poscon_list = [pos for pos in keep_poscon if pos > 0] 
                    dict_40['DMSO 0.0 10.0'] = keep_poscon_list
            if 'AgNP 100' in cmp:
                dict_100[cmp] = keep
                if negcon:
                    dict_100['Non-treated 0 0.0'] = keep_negcon
                if poscon:
                    keep_poscon_list = [pos for pos in keep_poscon if pos > 0] 
                    dict_100['DMSO 0.0 10.0'] = keep_poscon_list

        main_dict.append(dict_40)
        main_dict.append(dict_100)

    return main_dict

def df_x_y(main_dict):
    """
    From the list of dictionaries, get the x, y, x_log and label and save it to a dataframe
    """
    df_dict = []
    for dic in main_dict:
        lists = sorted(dic.items())
        x, y = zip(*lists)
        #get values from str(x) and transform it to float and log
        x_vals = []
        x_log = []
        for j in range(len(x)):
            if 'Non' in x[j][:3]: #for non-treated cells, consider a really low concentration instead of 0.0
                num = 0.00002
            elif 'DMS' in x[j][:3]:
                num = 10.0
            else:
                num = float(x[j][-6:])
            x_vals.append(num)
            x_log.append(np.log(num))
        #replicate the x values depending on the lenght of the y, from how many y values we have
        x_col = []
        x_log_col = []
        for lst in range(len(y)):
            l = len(y[lst])
            for n in range(l):
                x_col.append(x_vals[lst])
                x_log_col.append(x_log[lst])
        #create y column
        y_col = [item for sublist in y for item in sublist]
        #create dataframe
        df = pd.DataFrame({'x':x_col, 'x_log': x_log_col, 'y':y_col, 'label': x[0][:8]})
        df_dict.append(df)
    
    return df_dict

def df_list_plot(df_dict, plot_joined_replicates=False):
    """
    Defines the df to plot depending if the user wants to join all replicates (plot per-assay) or per plate basis.
    *df_dict is a list containing the df's with x, y, x_log to plot
    *plot_joined_replicates = True if want to plot all assays and ignore the plates
    """
    temp = []
    for dic in df_dict:
        temp.append(dic)
    joined_df = pd.concat(temp, axis='index', ignore_index=True)
    if plot_joined_replicates:
        mask40 = joined_df['label'].str.contains('40', case=False, na=False)
        mask100 = joined_df['label'].str.contains('100', case=False, na=False)
        df40 = joined_df[mask40].reset_index()
        df100 = joined_df[mask100].reset_index()
        df_list = [df40, df100]
    else:
        df_list = df_dict.copy()

    return df_list

def plot_linear_regression(df_list):
    """
    Plot linear regression curve for 40 and 100 nm AgNP
    *df_list is a list with dataframes for each plate or assay
    """
    for dic in df_list:
        if '40' in dic['label'][0]:
            p = sns.jointplot(x="x", y="y", data=dic,
                    kind="reg", truncate=False,
                    xlim=(-0.1, 0.3), ylim=(-1, 1),
                    color="m", height=7)
            p.fig.suptitle(dic.label[0], horizontalalignment='left')
        else:
            p = sns.jointplot(x="x", y="y", data=dic,
                    kind="reg", truncate=False,
                    xlim=(-0.2, 1.2), ylim=(-1, 1),
                    color="m", height=7)
            p.fig.suptitle(dic.label[0], horizontalalignment='left')
    return 

def dose_response_generator(df_list, define_bottom = False, bottom = None):
    """
    Generate the dose response curves based on a given list of dataframes
    *df_list is a list of dataframes
    """
    for df in df_list:
        #self-starter parameters
        logec50 = log_ec50(0.1, 1, 50) #calculate an estimative value for logec50
        p0 = [1, min(df.y), max(df.y), logec50] #initial guess for b slope, bottom c, top d and logec50 initial guess
        res, pcov = curve_fit(
            f=logistic_curve,
            xdata=df.x_log,
            ydata=df.y,
            p0=p0, 
            maxfev=10000
        )
        b, c, d, logec = res # this is the logistic curve fit
        if define_bottom: #change bottom to the given value 
            res[1] = float(bottom) 
            res[2] = float(1)
        #calculate R-squared
        modelPredictions = logistic_curve(df.x_log, *res)
        absError = modelPredictions - df.y
        SE = np.square(absError) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(df.y))
        print('R-squared:', Rsquared)
        #find the ec50 from the logec50 value
        ec50 = np.exp(logec)
        print('The EC50 is', round(ec50,3), 'micromolar or', round((ec50/100), 4), 'mg/mL')
        #plot
        plt.subplots(figsize=(13, 7))
        xvals = np.linspace(df.x_log.min(), df.x_log.max(), 100)
        sns.scatterplot(x="x_log", y="y", data=df, s=90)
        plt.plot(
            xvals,
            logistic_curve(xvals, *res),
            c="blue",
                label=f"Logistic Curve: Y = {round(c, 1)} + ({round(d, 1)} - {round(c, 1)})/(1 + 10**(({round(logec, 1)} - x)*{round(b, 1)})))",
        )
        plt.axhline(d, c="gray", linestyle="--", label="Upper Asymptote")
        plt.axhline(c, c="black", linestyle="--", label="Lower Asymptote")

        plt.title("Logistic Curve"), plt.legend(loc=1), plt.ylim(
            (df.y.min() - 0.2, df.y.max() + 0.1)
        );
        plt.show()
        #save this information to the df
        df['R-squared'] = Rsquared
        df['EC50 micromolar'] = round(ec50,3)
        df['EC50 mg/mL'] = round((ec50/100), 4)
    return

def prepare_plot_sm(df):
    plates = df['Metadata_Plate'].unique().tolist()
    df['Cells_CompoundCnc'] = df['Metadata_Compound'] + ' ' + df['Metadata_Concentration'].astype(str) + ' uM'
    list_cmp = df['Cells_CompoundCnc'].unique()
    df_corr = corr_matrix_per_plate(df, plates, plot_joined_replicates=True, metadata_column = 'Cells_CompoundCnc')
    plot_corr_matrix(df_corr, labelsize=7.5)

    return