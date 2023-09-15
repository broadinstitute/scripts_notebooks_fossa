import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from statannotations.Annotator import Annotator

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



def plot_and_stat(df, feature, x, palette, order_to_plot = [], hue_order_boxplot = [], pairs_stat = []):
    """
    
    """
    pval = normaltest(df[feature].values).pvalue
    if(pval < 0.05):
        print(f"Not normal distribution, pval equal {pval}. Using Mann-Whitney.")
        stat_test = "Mann-Whitney"
    else:
        print(f"Normal distribution, pval equal {pval}. Using independent t-test.")
        stat_test = "t-test_ind"
    df[feature].plot(kind='hist', title=feature)
    plt.show()

    sns.set(font_scale=1.6)
    sns.set_style("dark")
    g = sns.catplot(x=x,
                    y=feature,
                    hue="Metadata_Time",
                    kind="box",
                    legend=False,
                    height=6,
                    aspect=3,
                    palette=palette,
                    # boxprops={'alpha': 0.4},
                    data=df,
                    order=order_to_plot,
                    hue_order=hue_order_boxplot,
                    saturation=1
                    )
    sns.stripplot(data=df, x=x,
                    y=feature,
                    hue='Metadata_Plate',
                    jitter=True,
                    dodge=False,
                    marker='o',
                    linewidth=1,
                    palette="Paired",
                    # alpha=0.5,
                    order=order_to_plot,
                    size=15
                    )
    plt.xticks(rotation=60)
    plt.legend([],[], frameon=False)
    g.add_legend(title='Time (days)')
    g.set(xlabel=None, title=feature)

    annot = Annotator(g.ax, pairs_stat, 
                    data=df, x=x, y=feature, order=order_to_plot)
    annot.reset_configuration()
    annot.new_plot(g.ax, pairs_stat, data=df, x=x, y=feature, order=order_to_plot)
    annot.configure(test=stat_test, text_format='star', loc='inside', verbose=2).apply_test().annotate()

    plt.show()