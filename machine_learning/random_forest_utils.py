import pycytominer
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests



def lightgbm_iterations(df_, target="", ccp=0.08, n_estimators=10, max_depth=5,
                         slice=False, column_slice=None, slice_to_value=None, number_iterations=None):
    """
    This function takes a df (which you can slice based on a column and a value), creates a train/test, and trains a LightGBM Classifier model.
    We evaluate the model using confusion matrix, cross-validation, and shap.
    *df_ (DataFrame): dataframe that contains X and y
    *target (str): column name that will be the target (what are the classes?)
    *slice (bool): True if wants to slice the df based on a column_slice (str) and a variable that is in that column (slice_to_value) 

    return:
    shap_values_t: a list with the shap_values for all the features in the df for each class and sample;
    X_train: the portion of the dataframe that model was trained on.
    """
    import lightgbm as lgb
    from sklearn import metrics
    
    if slice:
        df = df_[df_[column_slice] == slice_to_value].reset_index(drop=True)
        print(f"Looping through {column_slice} = {slice_to_value}.")
    else:
        df = df_.copy()

    # features and metadata lists of cols
    feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    # X, y, and y target
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)
    y_target = y[target]

    all_feature_importances = []
    all_train_accuracies = []
    all_test_accuracies = []
    for _ in range(number_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y_target, random_state=42)
        # lgb_params = {
        #     'objective': 'multiclass',
        #     'num_class': len(y_target.unique()),  # Number of classes
        #     # 'metric': 'multi_logloss',
        #     # 'boosting_type': 'gbdt',
        #     # 'num_leaves': 31,
        #     # 'learning_rate': 0.05,
        #     # 'feature_fraction': 0.9,
        #     # 'bagging_fraction': 0.8,
        #     # 'bagging_freq': 5,
        #     # 'verbose': 0,
        #     'min_data_in_bin': 5,  # Adjust this value
        #     'min_data_in_leaf': 10,
        # }
        lgb_params = {
            'min_data_in_bin': 2,  # Adjust this value
            'min_data_in_leaf': 5,  # Adjust this value
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=0, n_estimators=n_estimators, max_depth=max_depth)
        lgb_model.fit(X_train, y_train)
        # Get feature importances for this iteration
        iteration_feature_importances = lgb_model.feature_importances_

        # Store the feature importances
        all_feature_importances.append(iteration_feature_importances)

        # Predictions on training set
        y_train_pred = lgb_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        all_train_accuracies.append(train_accuracy)

        # Predictions on test set
        y_test_pred = lgb_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        all_test_accuracies.append(test_accuracy)
    # Aggregate feature importances across iterations
    aggregate_feature_importances = np.mean(all_feature_importances, axis=0)

    # Rank features based on aggregated importances
    feature_ranking = np.argsort(aggregate_feature_importances)[::-1]

    # Calculate mean accuracy for training and testing sets
    mean_train_accuracy = np.mean(all_train_accuracies)
    mean_test_accuracy = np.mean(all_test_accuracies)

    print(f"\nMean Training Accuracy: {mean_train_accuracy}")
    print(f"Mean Testing Accuracy: {mean_test_accuracy}")

    return X, all_feature_importances, aggregate_feature_importances, lgb_model

def random_forest_iterations(df_, target="", ccp=0.08, n_estimators=10, max_depth=5,
                              slice=False, column_slice=None, slice_to_value=None, number_iterations=None,
                              top_n_features=50):
    """
    This function takes a df (which you can slice based on a column and a value), creates a train/test, and trains a
    Random Forest Classifier model. We evaluate the model using confusion matrix, cross-validation, and shap.
    
    Parameters:
    - df_ (DataFrame): DataFrame that contains X and y.
    - target (str): Column name that will be the target (what are the classes?).
    - slice (bool): True if you want to slice the df based on a column_slice (str) and a variable that is in that column (slice_to_value).
    - column_slice (str): Name of the column to slice.
    - slice_to_value: Value for slicing.
    - number_iterations: Number of iterations for training the model.
    - top_n_features: Number of top features to plot.

    Returns:
    - df_results: DataFrame containing 'features', 'importance', and 'importance_sd'.
    - X_train: The portion of the dataframe that the model was trained on.
    - forest: Trained Random Forest model.
    """
    if slice:
        df = df_[df_[column_slice] == slice_to_value]
        print(f"Looping through {column_slice} = {slice_to_value}.")
    else:
        df = df_.copy()

    # Features and metadata lists of columns
    feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)
    # X, y, and y target
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)
    y_target = y[target]

    all_feature_importances = []
    all_train_accuracies = []
    cv_scores_all = []

    for _ in range(number_iterations):
        # Initialize the model inside the loop to avoid potential leakage
        forest = RandomForestClassifier(random_state=0, ccp_alpha=ccp, n_estimators=n_estimators, max_depth=max_depth)

        # Train the model
        forest.fit(X, y_target)

        # Get feature importances for this iteration
        iteration_feature_importances = forest.feature_importances_

        # Store the feature importances
        all_feature_importances.append(iteration_feature_importances)

        # Perform cross-validation
        cv_scores = cross_val_score(forest, X, y_target, cv=5, scoring='accuracy')  # 5-fold cross-validation
        mean_cv = np.mean(cv_scores) * 100
        cv_scores_all.append(mean_cv)

        # Calculate mean accuracy for the training set
        mean_train_accuracy = accuracy_score(y_target, forest.predict(X))
        all_train_accuracies.append(mean_train_accuracy)

    # Aggregate feature importances across iterations
    aggregate_feature_importances = np.mean(all_feature_importances, axis=0)

    # Calculate standard deviation of feature importances across iterations
    feat_importance_sd = np.std(all_feature_importances, axis=0)

    # Rank features based on aggregated importances
    feature_ranking = np.argsort(aggregate_feature_importances)[::-1]

    # Create DataFrame with feature information
    features = [X.columns[feature_index] for feature_index in feature_ranking]
    importance = aggregate_feature_importances[feature_ranking]
    importance_sd = feat_importance_sd[feature_ranking]
    
    mean_cv_accuracy = np.mean(cv_scores_all)
    print(f"Cross-Validation Accuracy mean: {mean_cv_accuracy}%")

    # Calculate mean accuracy for the training set
    mean_train_accuracy = np.mean(all_train_accuracies)
    print(f"\nMean Training Accuracy: {mean_train_accuracy}")

    df_results = pd.DataFrame({
        'features': features,
        'importance': importance,
        'importance_sd': importance_sd
    })

    # Plot the top features
    fig, ax = plt.subplots()
    df_results.head(top_n_features).sort_values(by=['importance'], ascending=True).plot.barh(x='features', y='importance', yerr='importance_sd', ax=ax, align="center")
    ax.set_title(f"Top {top_n_features} Feature importances ({number_iterations} iterations)")
    ax.set_xlabel("Feature importance")
    fig.set_size_inches(18.5, 10.5)
    plt.show()

    
    return df_results, X, forest

def random_forest_model_eval(df_, target = "", ccp = 0.08, n_estimators=10, max_depth=5, 
                             slice = False, column_slice = None, slice_to_value = None,
                            ):
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

def most_important_with_sd(X, all_feature_importances, aggregate_feature_importances, number_of_features_select=30, compound=None):
    """
    """
    feat_importance_sd = np.std(all_feature_importances, axis=0)
    feature_ranking = np.argsort(aggregate_feature_importances)[::-1]
    features=[]
    importance=[]
    importance_sd=[]
    for i, feature_index in enumerate(feature_ranking):
            if i < number_of_features_select:
                    features.append(X.columns[feature_index])
                    importance.append(aggregate_feature_importances[feature_index])
                    importance_sd.append(feat_importance_sd[feature_index])
    
    df_results = pd.DataFrame(list(zip(features, importance, importance_sd)), columns=[f'{compound}_features', f'{compound}_importance', f'{compound}_importance_sd'])

    fig, ax = plt.subplots()
    df_results.sort_values(by=[f'{compound}_importance'],ascending=True).plot.barh(x=f'{compound}_features', y=f'{compound}_importance', yerr=f'{compound}_importance_sd', ax=ax,align="center")
    ax.set_title(f"{compound} feature importances (100 iterations)")
    ax.set_xlabel("Feature importance")
    fig.set_size_inches(18.5, 10.5)
    plt.show()

    return df_results

def xgboost(df_, target="", slice=False, column_slice=None, slice_to_value=None, params=None, number_iterations=1):
    all_feature_importances = []
    all_train_accuracies = []
    all_test_accuracies = []

    for _ in range(number_iterations):
        if slice:
            df = df_[df_[column_slice] == slice_to_value].reset_index(drop=True)
            print(f"Looping through {column_slice} = {slice_to_value}.")
        else:
            df = df_.copy()

        # features and metadata lists of cols
        feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
        meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)

        # X, y, and y target
        X = pd.DataFrame(df, columns=feat)
        y = pd.DataFrame(df, columns=meta)
        y_target = y[target]

        # Check unique values in the target column
        unique_labels = y_target.unique()
        print(f"Unique labels in target column: {unique_labels}")

        # Check data types of the target column
        print(f"Data type of target column: {y_target.dtype}")

        # Check if it's binary classification
        num_classes = len(unique_labels)
        print(f"Number of classes in target column: {num_classes}")

        # Confirm binary classification
        if num_classes != 2:
            raise ValueError("Binary classification requires exactly two classes in the target variable.")

        # Convert boolean to integer if needed
        if y_target.dtype == bool:
            y_target = y_target.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y_target, random_state=42)
        print("Class Distribution:")
        print(y_train.value_counts())

        # Create regression matrices
        dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

        results = xgb.cv(
            params, dtrain_clf,
            num_boost_round=2000,
            nfold=5,
            metrics=["logloss", "auc", "error"],
        )

        # Get the best number of boosting rounds
        num_boost_rounds = len(results)
        print(f'Best number of boosting rounds: {num_boost_rounds}')

        # Train the final model with the best number of boosting rounds
        bst = xgb.train(params, dtrain_clf, num_boost_round=num_boost_rounds)

        # Make predictions on the test set
        y_pred_probs = bst.predict(dtest_clf)
        y_pred_labels = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

        # evaluate predictions
        accuracy = accuracy_score(y_test, y_pred_labels)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # Get feature importance scores
        importance = bst.get_fscore()

        # Sort features by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Extract features and importance scores
        features, importance_scores = zip(*sorted_importance)

        # Store feature importances and accuracies for each iteration
        all_feature_importances.append(importance_scores)
        all_train_accuracies.append(accuracy_score(y_train, bst.predict(dtrain_clf) > 0.5))
        all_test_accuracies.append(accuracy_score(y_test, y_pred_labels))

    # Aggregate feature importances across iterations
    aggregate_feature_importances = np.mean(all_feature_importances, axis=0)

    # Rank features based on aggregated importances
    feature_ranking = np.argsort(aggregate_feature_importances)[::-1]

    # Calculate mean accuracy for training and testing sets
    mean_train_accuracy = np.mean(all_train_accuracies)
    mean_test_accuracy = np.mean(all_test_accuracies)

    print(f"\nMean Training Accuracy: {mean_train_accuracy}")
    print(f"Mean Testing Accuracy: {mean_test_accuracy}")

    return X, all_feature_importances, aggregate_feature_importances, bst

def xgboost_grid_search(df_, target="", slice=False, column_slice=None, slice_to_value=None, param_grid=None):
    if slice:
        df = df_[df_[column_slice] == slice_to_value].reset_index(drop=True)
        print(f"Looping through {column_slice} = {slice_to_value}.")
    else:
        df = df_.copy()

    # features and metadata lists of cols
    feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)

    # X, y, and y target
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)
    y_target = y[target]

    # Check unique values in the target column
    unique_labels = y_target.unique()
    print(f"Unique labels in target column: {unique_labels}")

    # Check data types of the target column
    print(f"Data type of target column: {y_target.dtype}")

    # Check if it's binary classification
    num_classes = len(unique_labels)
    print(f"Number of classes in target column: {num_classes}")

    # Confirm binary classification
    if num_classes != 2:
        raise ValueError("Binary classification requires exactly two classes in the target variable.")

    # Convert boolean to integer if needed
    if y_target.dtype == bool:
        y_target = y_target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_target, random_state=42)

    # Create regression matrices
    dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # Perform grid search
    xgb_model = XGBClassifier(objective='binary:logistic', seed=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Print best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Train the final model with the best parameters
    best_params = grid_search.best_params_
    final_model = XGBClassifier(objective='binary:logistic', seed=42, **best_params)
    final_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = final_model.predict(X_test)

    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on Test Set:", accuracy)

    return

def rank_features_by_ttest(df_, target_variable, slice=False, column_slice=None, slice_to_value=None):
    """
    Perform t-test for each feature with respect to a target variable, considering optional data slicing,
    and rank features based on corrected p-values.

    Parameters:
    - df_: DataFrame containing features and the target variable.
    - target_variable: The target variable for which t-tests will be performed.
    - feature_names: List of feature names to consider.
    - slice: Boolean indicating whether to slice the DataFrame based on a column.
    - column_slice: Column name for slicing (used if 'slice' is True).
    - slice_to_value: Value for slicing (used if 'slice' is True).

    Returns:
    - results_df: DataFrame containing feature names, uncorrected p-values, and corrected p-values, sorted by corrected p-values.
    """
    if slice:
        df = df_[df_[column_slice] == slice_to_value].copy()
        print(f"Looping through {column_slice} = {slice_to_value}.")
    else:
        df = df_.copy()

    feature_names = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)

    feature_ranking = []

    for feature in feature_names:
        group1 = df[df[target_variable] == 0][feature]
        group2 = df[df[target_variable] == 1][feature]

        # Check if there is variability in the values
        if len(set(group1).union(set(group2))) > 1:
            t_stat, p_value = ttest_ind(group1, group2)
            feature_ranking.append((feature, p_value))
        else:
            feature_ranking.append((feature, float('nan')))

    # Extract p-values for correction
    p_values = [p_value for feature, p_value in feature_ranking]

    # Correct p-values using FDR correction
    _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    # Update the feature ranking with corrected p-values
    feature_ranking = [(feature, uncorrected_p_value, corrected_p_value)
                       for (feature, uncorrected_p_value), corrected_p_value in zip(feature_ranking, corrected_p_values)]

    # Sort features by corrected p-value in ascending order
    feature_ranking.sort(key=lambda x: x[2])

    # Create a DataFrame from the results
    results_df = pd.DataFrame(feature_ranking, columns=['features', 'p_value', 'p_value_fdr'])

    return results_df

def logreg(df_, target="", slice=False, column_slice=None, slice_to_value=None, top_n_features=50):
    """
    Train a logistic regression model and return relevant information.

    Parameters:
    - df_: DataFrame containing features and the target variable.
    - target: The target variable for logistic regression.
    - slice: Boolean indicating whether to slice the DataFrame based on a column.
    - column_slice: Column name for slicing (used if 'slice' is True).
    - slice_to_value: Value for slicing (used if 'slice' is True).

    Returns:
    - results_df: DataFrame containing 'feature', feature importance, and normalized feature importance, sorted by importance.
    - X_train: Training features.
    - y_train: Training target variable.
    - normalized_feature_importance: Feature importance normalized for better interpretation.
    - model: Trained logistic regression model.
    """
    if slice:
        df = df_[df_[column_slice] == slice_to_value].reset_index(drop=True)
        print(f"Looping through {column_slice} = {slice_to_value}.")
    else:
        df = df_.copy()

    # Load your data (replace X and Y with your features and labels)
    # X should be a 2D array (samples x features), and Y should be a 1D array (labels)
    # Example:
    # features and metadata lists of cols
    feat = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=False)
    meta = pycytominer.cyto_utils.features.infer_cp_features(df, metadata=True)

    # X, y, and y target
    X = pd.DataFrame(df, columns=feat)
    y = pd.DataFrame(df, columns=meta)
    y_target = y[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)

    # Initialize the logistic regression model
    model = LogisticRegression(C=100)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model performance (you can use other metrics as needed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
    print(f"Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%")

    # Extract feature importance
    feature_importance = np.abs(model.coef_[0])  # Taking the absolute values for importance
    normalized_feature_importance = feature_importance / np.sum(feature_importance)  # Normalize for better interpretation

    # Create a DataFrame with feature information
    results_df = pd.DataFrame({
        'features': feat,
        'importance': feature_importance,
        'importance_normalized': normalized_feature_importance
    })

    # Sort DataFrame by 'normalized_feature_importance' in descending order
    results_df = results_df.sort_values(by='importance_normalized', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots()
    results_df.head(top_n_features).sort_values(by=['importance_normalized'], ascending=True).plot.barh(x='features', y='importance_normalized', ax=ax, align="center")
    ax.set_title(f"Top {top_n_features} Feature importances LogReg)")
    ax.set_xlabel("Feature importance")
    fig.set_size_inches(18.5, 10.5)
    plt.show()

    return results_df, X_train, y_train, normalized_feature_importance, model

def print_features_above_threshold(model, X, y, importance_threshold=0, random_state=42, n_permutations=100):
    """
    Print features with permutation importances above a specified threshold.

    Parameters:
    - model: The trained model for which permutation importances will be calculated.
    - X: Input features.
    - y: Target labels.
    - importance_threshold: Threshold for feature importance values.
    - random_state: Random seed for reproducibility.
    - n_permutations: Number of permutations to perform.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Calculate permutation importances
    perm_importances = permutation_importance(model, X_test, y_test, n_repeats=n_permutations, random_state=random_state)

    # Get feature names
    feature_names = X.columns

    # Filter features based on the importance threshold
    important_features = [(feature, importance) for feature, importance in zip(feature_names, perm_importances.importances_mean) if importance > importance_threshold]

    # Sort features by importance value
    sorted_features = sorted(important_features, key=lambda x: x[1], reverse=True)

    # Print sorted features
    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")

    return feature_names, sorted_features

############################ FUNCTIONS TO FIX LATER #####################################

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