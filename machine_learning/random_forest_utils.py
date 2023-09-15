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