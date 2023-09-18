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