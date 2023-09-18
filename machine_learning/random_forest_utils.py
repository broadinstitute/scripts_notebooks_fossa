import pycytominer
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def random_forest_model_eval(df_, target = "", ccp = 0.08, n_estimators=10, max_depth=5, slice = False, column_slice = None, slice_to_value = None):
    """
    This function takes a df (which you can slice based on a column and a value), creates a train/test, and train a Random Forest Classifier model.
    We evaluate the model using confusion matrix, cross validation, and shap.
    *df_ (DataFrame): dataframe that contains X and y
    *target (str): column name that will be the target (what are the classes?)
    *slice (bool): True if wants to slice the df based on a column_slice (str) and a variable that is in that column (slice_to_value) 

    return:
    shap_values_t: a list with the shap_values for all the features in the df for each class and sample;
    X_train: the portion of the dataframe that model was trained on.
    """
    if slice:
        df = df_[df_[column_slice] == slice_to_value]
        print(f"Looping through {column_slice} = {slice_to_value}.")
    else:
        df = df_.copy()
    
    #features and metadata lists of cols
    feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    #X, y and y target    
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)
    y_target = y[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, random_state=42)
    #train
    forest = RandomForestClassifier(random_state=0, ccp_alpha=ccp, n_estimators=n_estimators, max_depth=max_depth)
    forest.fit(X_train, y_train)
    #evaluate
    scores = cross_val_score(forest, X_train, y_train, cv=5)
    print(f"All the scores from cross_val_score: {scores}")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print('Classic: Accuracy of Decision Tree classifier on training set: {:.2f}'
            .format(forest.score(X_train, y_train)))
    print('Classic: Accuracy of Decision Tree classifier on test set: {:.2f}'
        .format(forest.score(X_test, y_test)))
    #create the confusion matrix
    y_pred = forest.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=forest.classes_).plot()
    plt.show()
    
    #summary shap
    explainer = shap.TreeExplainer(forest)
    shap_values_t = explainer.shap_values(X_train)
    shap.summary_plot(shap_values_t, X_train, plot_type="bar", class_names=forest.classes_, max_display=10)
    plt.show()
    
    #beeswarm plot shap
    shap_values_ind = explainer(X_train)
    for i in range(len(forest.classes_)):
        shap.plots.beeswarm(shap_values_ind[:,:,i], max_display=10,show=False)
        plt.title(f"Class: {forest.classes_[i]}")
        plt.show()

    #decision plot shap
    row_index=2
    base = explainer.expected_value
    shap.multioutput_decision_plot(list(base), shap_values_t,
                               row_index=row_index, 
                               feature_names=X_train.columns.to_list(), 
                               )
    
    return shap_values_t, X_train, forest
                
def loop_random_forest_model_eval(df_, target = "", column_to_loop = "", list_to_loop = [], ccp = 0.08, n_estimators = 10, max_depth = 5):
    """
    This function takes a df (which you can slice based on a column and a value), creates a train/test, and train a Random Forest Classifier model.
    We evaluate the model using confusion matrix, cross validation, and shap.
    *df_ (DataFrame): dataframe that contains X and y
    *target (str): column name that will be the target (what are the classes?)
    *slice (bool): True if wants to slice the df based on a column_slice (str) and a variable that is in that column (slice_to_value) 

    return:
    shap_values_t: a list with the shap_values for all the features in the df for each class and sample;
    X_train: the portion of the dataframe that model was trained on.
    """
    for value in list_to_loop:
        df = df_[df_[column_to_loop] == value]
        print(f"Looping through {column_to_loop} = {value}.")
        #features and metadata lists of cols
        feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
        meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
        #X, y and y target    
        X = pd.DataFrame(df, columns=feat)
        y = pd.DataFrame(df, columns=meta)
        y_target = y[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y_target, random_state=42)
        #train
        forest = RandomForestClassifier(random_state=0, ccp_alpha=ccp, n_estimators=n_estimators, max_depth=max_depth)
        forest.fit(X_train, y_train)
        #evaluate
        scores = cross_val_score(forest, X_train, y_train, cv=5)
        print(f"All the scores from cross_val_score: {scores}")
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print('Classic: Accuracy of Decision Tree classifier on training set: {:.2f}'
                .format(forest.score(X_train, y_train)))
        print('Classic: Accuracy of Decision Tree classifier on test set: {:.2f}'
            .format(forest.score(X_test, y_test)))
        #create the confusion matrix
        y_pred = forest.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=forest.classes_).plot()
        plt.show()
        
        #summary shap
        explainer = shap.TreeExplainer(forest)
        shap_values_t = explainer.shap_values(X_train)
        shap.summary_plot(shap_values_t, X_train, plot_type="bar", class_names=forest.classes_, max_display=10)
        plt.show()
        
        #beeswarm plot shap
        shap_values_ind = explainer(X_train)
        for i in range(len(forest.classes_)):
            shap.plots.beeswarm(shap_values_ind[:,:,i], max_display=10,show=False)
            plt.title(f"Class: {forest.classes_[i]}")
            plt.show()

        #decision plot shap
        row_index=2
        base = explainer.expected_value
        shap.multioutput_decision_plot(list(base), shap_values_t,
                                row_index=row_index, 
                                feature_names=X_train.columns.to_list(), 
                                )
        
############################ TO FIX LATER

def pruning():
    """
    comes from https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    """
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()

def eval_classic():
    """
    #good explanation at https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
    """
    

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

def plot_one_tree():
    """
    1. Feature name + split value. Split value — split value is decided after selecting a threshold value which gives highest information gain for that split.

    2. Gini impurity: 1 - (probability of Yes)^2 - (probability of No)^2.
        - Gini — It is basically deciding factor i.e. to select feature at next node , to pick best split value etc.

    3. Samples — No of samples remaining at that particular node.

    4. Values — No of samples of each class remaining at that particular node.
    """

    from sklearn import tree
    plt.figure(figsize=(12,12))
    tree.plot_tree(forest.estimators_[0], feature_names=X.columns, filled=True, class_names=forest.classes_.astype('str').tolist(),rounded=True, fontsize=10)