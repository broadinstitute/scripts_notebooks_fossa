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



def plot_and_stat(df, feature, x, palette, order_to_plot = [], hue_order_boxplot = [], 
                  pairs_stat = [], show_hist=False, rotation=75, 
                  new_labels=[], ylabel="", hue_col="Metadata_Time",
                  title_legend='Time (days)',
                  set_lim=False, xlim=None, ylim=None):
    """
    
    """
    pval = normaltest(df[feature].values).pvalue
    if (pval < 0.05):
        print(f"Not normal distribution, pval equal {pval}. Using Mann-Whitney.")
        stat_test = "Mann-Whitney"
    else:
        print(f"Normal distribution, pval equal {pval}. Using independent t-test.")
        stat_test = "t-test_ind"
    if show_hist:
        df[feature].plot(kind='hist', title=feature)
        plt.show()

    sns.set(font_scale=1.8)
    sns.set_style("dark")
    g = sns.catplot(x=x,
                    y=feature,
                    hue=hue_col,
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
                    jitter=False,
                    dodge=False,
                    marker='o',
                    linewidth=1,
                    palette="Paired",
                    # alpha=0.5,
                    order=order_to_plot,
                    size=8
                    )
    plt.xticks(rotation=rotation)
    plt.legend([],[], frameon=False)
    g.add_legend(title=title_legend)
    g.set(xlabel=None, title=feature, ylabel=ylabel)
    if set_lim:
        g.ax.set_ylim(xlim,ylim)

    annot = Annotator(g.ax, pairs_stat, 
                    data=df, x=x, y=feature, order=order_to_plot)
    annot.reset_configuration()
    annot.new_plot(g.ax, pairs_stat, data=df, x=x, y=feature, order=order_to_plot)
    annot.configure(test=stat_test, text_format='star', loc='inside', verbose=2).apply_test().annotate()

    g.set_xticklabels(new_labels, fontsize=10)

    plt.show()

def plot_with_markers(df, feature, x, palette_boxplot=None, order_to_plot = [], hue_order_boxplot = [], 
                  pairs_stat = [], show_hist=False, rotation=75, 
                  new_labels=[], ylabel="", hue_col="Metadata_Time", title_plot=None,
                  title_legend='Time (days)',
                  set_lim=False, xlim=None, ylim=None,
                  col_groupby=None, category_mapping=None,
                  perform_stat_test=True):
    """
    
    """    
    # pval = normaltest(df[feature].values).pvalue
    # if (pval < 0.05):
    #     print(f"Not normal distribution, pval equal {pval}. Using Mann-Whitney.")
    #     stat_test = "Mann-Whitney"
    # else:
    #     print(f"Normal distribution, pval equal {pval}. Using independent t-test.")
    #     stat_test = "t-test_ind"
    # if show_hist:
    #     df[feature].plot(kind='hist', title=feature)
    #     plt.show()
    stat_test = "t-test_ind"
    
    sns.set(font_scale=1.7)
    sns.set_style("dark")
    g=sns.catplot(data=df,
                  x=x,
                  y=feature,
                  kind="box",
                  legend=False,
                  height=7,
                  aspect=1.5,
                  palette=palette_boxplot,
                  boxprops={'alpha': 0.6},
                  order=order_to_plot,
                  saturation=1
                  )

    # Create the strip plot
    for category, group_data in df.groupby(col_groupby):
        properties = category_mapping.get(category, {"marker": "o", "color": "black"})
        marker = properties["marker"]
        color = properties["color"]
        plt.scatter(group_data[x], group_data[feature], label=category, marker=marker, 
                    color="none",  # Set facecolor to "none" for no fill
            edgecolor=color,  # Use the specified color for marker edges
                    )
    plt.legend([],[], frameon=False)
    g.add_legend(title='')
    g.set(xlabel=None, title=title_plot, ylabel=ylabel)
    # Customize the appearance
    if set_lim:
        g.ax.set_ylim(xlim,ylim)
    plt.xticks(rotation=rotation)
    plt.legend([],[], frameon=False)
    # g.add_legend(title=title_legend)

    if perform_stat_test:
        annot = Annotator(g.ax, pairs_stat, 
                            data=df, x=x, y=feature, order=order_to_plot)
        annot.reset_configuration()
        annot.new_plot(g.ax, pairs_stat, data=df, x=x, y=feature, order=order_to_plot)
        annot.configure(test=stat_test, text_format='star', loc='inside', verbose=2).apply_test().annotate()
    g.set_xticklabels(new_labels, fontsize=15)
    plt.show()