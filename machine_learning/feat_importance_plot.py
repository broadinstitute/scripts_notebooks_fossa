import matplotlib.pyplot as plt


def get_lists_plot(features, compartments, groups, channels, title, cmpt_pallete, groups_pallete, channel_pallete):

    #get lists
    feat_per_cmp = []
    for c in range(len(compartments)):
        temp = []
        for f in range(len(features)):
            if features[f][:5] in compartments[c]:
                temp.append(features[f])
        feat_per_cmp.append(temp)
    feat_per_cmp_per_group = []
    for g in groups:
        temp_group = []
        for lst in feat_per_cmp:
            temp = []
            # temp.append('Metadata_Compound_concentration')
            for feat in lst:
                if g in feat:
                    temp.append(feat)
            temp_group.append(temp)
        feat_per_cmp_per_group.append(temp_group)
    #get channels
    feat_per_channel = []
    for c in range(len(channels)):
        temp = []
        for f in range(len(features)):
            if channels[c] in features[f]:
                temp.append(features[f])
        feat_per_channel.append(temp)

    #get number of features per compartment
    number_feat_per_cmpt_lst = []
    for f in range(len(feat_per_cmp)):
        l = len(feat_per_cmp[f])
        number_feat_per_cmpt_lst.append(l)
    plt.pie(number_feat_per_cmpt_lst, labels=compartments, colors=cmpt_pallete, autopct='%.0f%%',)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    #get number of features per compartment
    number_feat_per_group_lst = []
    for f in range(len(feat_per_cmp_per_group)):
        l = sum(map(len, feat_per_cmp_per_group[f]))
        number_feat_per_group_lst.append(l)
    patches, texts, _ = plt.pie(number_feat_per_group_lst, labels=groups, colors=groups_pallete, autopct='%.0f%%',)
    plt.legend(patches, groups, bbox_to_anchor=(1,0), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    #get percentage of channels in the 
    number_feat_per_channel_lst = []
    for f in range(len(feat_per_channel)):
        l = len(feat_per_channel[f])
        number_feat_per_channel_lst.append(l)
    patches, texts, _ = plt.pie(number_feat_per_channel_lst, labels=channels, colors=channel_pallete, autopct='%.0f%%',)
    plt.legend(patches, channels, bbox_to_anchor=(1,0), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def feats_per_compartment_per_group_per_channel(compartments, features, groups, channels):
    """
    """
    #get list per compartments
    feat_per_cmp = []
    for c in range(len(compartments)):
        temp = []
        for f in range(len(features)):
            if features[f][:5] in compartments[c]:
                temp.append(features[f])
        feat_per_cmp.append(temp)
    #get list per group of feature
    feat_per_cmp_per_group = []
    for g in groups:
        temp_group = []
        for lst in feat_per_cmp:
            temp = []
            # temp.append('Metadata_Compound_concentration')
            for feat in lst:
                if g in feat.split("_")[1]:
                    temp.append(feat)
            temp_group.append(temp)
        feat_per_cmp_per_group.append(temp_group)
    #get channels
    feat_per_channel = []
    for c in range(len(channels)):
        temp = []
        for f in range(len(features)):
            if channels[c] in features[f]:
                temp.append(features[f])
        feat_per_channel.append(temp)

    return feat_per_cmp, feat_per_cmp_per_group, feat_per_channel

## NOTE: if the feature subgroup is "Correlation", both channels are being counted 
# (DNA_Mito, for example), so they will be in the list of DNa and also for Mito
def count_list_simple(simple_list, feat_number):
    """
    """
    number_feat_lst = []
    percent_lst = []
    for f in range(len(simple_list)):
        l = len(simple_list[f])
        number_feat_lst.append(l)
    
    total_feat = sum(number_feat_lst)
    for f in range(len(simple_list)):
        l = len(simple_list[f])
        p = (l*100)/total_feat
        percent_lst.append(p)
    
    return number_feat_lst, percent_lst

def count_list_of_sublists(complex_list, feat_number):
    """
    """
    number_feat_lst = []
    percent_lst = []
    for f in range(len(complex_list)):
        l = sum(map(len, complex_list[f]))
        number_feat_lst.append(l)
    
    total_feat = sum(number_feat_lst)
    for f in range(len(complex_list)):
        l = sum(map(len, complex_list[f]))
        p = (l*100)/total_feat
        percent_lst.append(p)

    return number_feat_lst, percent_lst

def plot_stacked_bar(df, x='Metadata_Time', ylabel="Percentage (%)", title="", colormap=None, rotation=45, percentage_fontsize=18):
        ax = df.plot(x=x, 
                     kind='bar', 
                     stacked=True,
                     color=colormap)
        plt.rcParams.update({'font.size': percentage_fontsize})
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1.0, 0.5), 
            reverse=True)
        plt.ylabel(ylabel, fontsize=20)
        plt.xlabel("")
        plt.xticks(rotation=rotation,fontsize=15)
        plt.title(title,fontsize=20)
        for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                # print(width, height)
                if not height == 0.0:
                        ax.text(x+width/2, 
                                y+height/2, 
                                '{:.0f}%'.format(height), 
                                horizontalalignment='center', 
                                verticalalignment='center')
        plt.show()

def plot_stacked_bar_horizontal(df, x='Metadata_Time', ylabel="Percentage (%)", title="", colormap=None, rotation=45, percentage_fontsize=18):
        ax = df.plot(x=x, 
                     kind='barh', 
                     stacked=True,
                     color=colormap)
        plt.rcParams.update({'font.size': percentage_fontsize})
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1.0, 0.5), 
            # reverse=True
            )
        plt.ylabel("")
        plt.xlabel(ylabel, fontsize=15)
        plt.xticks(rotation=rotation,fontsize=12)
        plt.title(title,fontsize=15)
        for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                if not width == 0.0:
                        ax.text(x+width/2, 
                                y+height/2, 
                                '{:.0f}%'.format(width), 
                                horizontalalignment='center', 
                                verticalalignment='center')
        plt.show()

def count_feats_3_panels(features):

    feat_per_panel = [[],[],[]]
    for f in features:
        if f.split("_")[-1] == "CP":
            feat_per_panel[0].append(f)
        elif f.split("_")[-1] == "LCP":
            feat_per_panel[1].append(f)
        elif f.split("_")[-1] == "TP":
            feat_per_panel[2].append(f)

    return feat_per_panel