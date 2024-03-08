import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

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
                  set_lim=False, xlim=None, ylim=None,
                  perform_stat_test=True, add_legend=True, fontsize_xticks=10,
                  plot_height=6, plot_aspect=3):
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

    sns.set(font_scale=1.9)
    sns.set_style("dark")
    g = sns.catplot(x=x,
                    y=feature,
                    hue=hue_col,
                    kind="box",
                    legend=False,
                    height=plot_height,
                    aspect=plot_aspect,
                    palette=palette,
                    boxprops={'alpha': 0.4},
                    # boxprops={'facecolor': 'white', 'edgecolor': palette},
                    medianprops=dict(color="black", alpha=1, linewidth=2),
                    data=df,
                    order=order_to_plot,
                    hue_order=hue_order_boxplot,
                    saturation=1,
                    dodge=False
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

    if add_legend:
        g.add_legend(title=title_legend)
    g.set(xlabel=None, title=feature, ylabel=ylabel)
    if set_lim:
        g.ax.set_ylim(xlim,ylim)

    if perform_stat_test:
        annot = Annotator(g.ax, pairs_stat, 
                        data=df, x=x, y=feature, order=order_to_plot)
        annot.reset_configuration()
        annot.new_plot(g.ax, pairs_stat, data=df, x=x, y=feature, order=order_to_plot)
        annot.configure(test=stat_test, text_format='star', loc='inside', verbose=2).apply_test().annotate()
    g.set_xticklabels(new_labels, fontsize=fontsize_xticks)
    fig = g.ax.get_figure()
    fig.savefig(f"{feature}.svg")
    plt.show()
    return

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


def plot_with_anova(df, feature, x, palette_boxplot=None, order_to_plot = [], hue_order_boxplot = [], 
                  pairs_stat = [], show_hist=False, rotation=75, 
                  new_labels=[], ylabel="", hue_col="Metadata_Time", title_plot=None,
                  title_legend='Time (days)',
                  set_lim=False, xlim=None, ylim=None,
                  col_groupby=None, category_mapping=None,
                  perform_stat_test=True):
    """
    
    """    
    from scipy.stats import f_oneway
    # Required descriptors for annotate
    custom_long_name = 'One-way ANOVA statistical test'
    custom_short_name = 'One-way ANOVA'
    custom_func = f_oneway
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
    
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
        annot.configure(test=custom_test, text_format='star', loc='inside', verbose=2).apply_test().annotate()
    g.set_xticklabels(new_labels, fontsize=15)
    plt.show()

def plot_pvalue_calculated_elsewhere(df, feature, x, palette_boxplot=None, order_to_plot = [], hue_order_boxplot = [], 
                  pairs_stat = [], show_hist=False, rotation=75, 
                  new_labels=[], ylabel="", hue_col="Metadata_Time", title_plot=None,
                  title_legend='Time (days)',
                  set_lim=False, xlim=None, ylim=None,
                  col_groupby=None, category_mapping=None,
                  perform_stat_test=True,
                  p_values=None):
    """
    
    """ 
    
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
        annot.configure(text_format="star", loc="inside")
        annot.set_pvalues_and_annotate(p_values)
    g.set_xticklabels(new_labels, fontsize=15)
    plt.show()


def z_test_pairs(df, feature, pairs, label_column):
    """
    """
    from statsmodels.stats.weightstats import ztest

    pvalues_list = []
    for p in pairs:
        ztest_value, p_valor = ztest(x1=df.query(f"{label_column} in '{p[0]}'").reset_index()[feature], x2=df.query(f"{label_column} in '{p[1]}'").reset_index()[feature])
        if p_valor<.05:
            print(f'{p[0]} x {p[1]}: p-value {p_valor}, REJECT null hypothesis')
        else:
            print(f"{p[0]} x {p[1]}: p-value {p_valor}, failed to reject null hypothesis")
        pvalues_list.append(p_valor)

    return pvalues_list