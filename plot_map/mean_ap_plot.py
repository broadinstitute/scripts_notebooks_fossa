import matplotlib.pyplot as plt
import seaborn as sns

def inputs_map():
    """
    Get names of root and plot titles to give as an input to plot_map function

    """
    import os 
    import easygui as eg

    # Path
    root = eg.diropenbox(title="Choose main path where folders are")
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
    print("List of directories", dirlist)  
    # Title box  
    text = "Enter the title for each corresponding plot"
    title = "Window Title"
    output = eg.multenterbox(text, title, dirlist)
    print("List of titles:", output)

    return root, dirlist, output

def plot_map(df, p_value=False, label_column="Metadata_Compound", hue_column="Metadata_Compound", title="mAP", significance_threshold=0.05, y_lim=(0.2, 4.0)):
    """
    Plot the mean Average Precision (mAP) after running evalzoo_matric. 
    *df (DataFrame): 
    The default is to plot q_value, but p_value can be plot by changing p_value to True.
     
    """
    
    from numpy import log10
    from adjustText import adjust_text

    # Define inputs
    x = df["sim_retrieval_average_precision_ref_i_mean_i"]

    if p_value:
        y = df["sim_retrieval_average_precision_ref_i_nlog10pvalue_mean_i"]
    else:
        y = df["sim_retrieval_average_precision_ref_i_nlog10qvalue_mean_i"]
    
    labels = df[label_column]
    hue = df[hue_column]
    title=title
    significance_threshold = significance_threshold

    # Set size and fonts
    sns.set(rc={'figure.figsize':(15,10)}, font_scale=1.5)

    # Plot
    g = sns.scatterplot(x=x, 
                        y=y,
                        hue=hue, 
                        s=70, 
                        legend=False)

    # Add horizontal line based on the significance_threshold
    g.axhline(y=-log10(0.05), linewidth=2.5, color='r')
    g.axes.set_title(title,fontsize=30)
    g.set_ylim(y_lim)
    g.set_xlim(0.2, 1.0)
    # plt.title(title)

    # Define texts for points and use adjust_text to find better position
    texts = [plt.text(x_pos, y_pos, f'{l}') for (x_pos, y_pos, l) in zip(x, y, labels)];
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

    plt.show()
    return 