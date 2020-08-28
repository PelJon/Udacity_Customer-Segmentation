from typing import List, Any
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from imblearn.ensemble import RUSBoostClassifier, EasyEnsembleClassifier


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        print(X.select_dtypes(include=['object']).columns.tolist())
        
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col].astype(str))
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
                
        check_subset_object = output.select_dtypes(include=['object']).columns.tolist()

        if not check_subset_object:
            print("Object transformation successful, no objects in dataset")
        
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def replace_missing_data(dataset):
    """
        Replace the missing values with NaN-Values

        Args:
            dataset (dataframe): The dataset to be split

        Returns:
            dataset: Processed dataset without wrongly inserted missing values
    """
    zero_unkown_list = ['ANZ_TITEL','D19_TELKO_ONLINE_QUOTE_12','HEALTH_TYP','KBA05_ANTG4','KBA05_KRSKLEIN','KBA05_KRSOBER','KBA05_KRSVAN','KBA05_KRSZUL','KBA05_MAXVORB','KBA13_ANTG4','KBA13_KMH_110','KBA13_KMH_251','KBA13_KW_30','NATIONALITAET_KZ','PLZ8_ANTG4','STRUKTURTYP', 'D19_TELKO_ONLINE_QUOTE_12', 'ANZAHL_TITEL', 'ALTER_HH', 'AGER_TYP', 'CJT_GESAMTTYP', 'HEALTH_TYP', 'WOHNDAUER_2008', 'KKK', 'W_KEIT_KIND_HH', 'ALTERSKATEGORIE_GROB', 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'KBA05_GBZ', 'TITEL_KZ', 'REGIOTYP', 'KBA05_BAUMAX', 'HH_EINKOMMEN_SCORE', 'GEBAEUDETYP', 'ANREDE_KZ']
    XX_unkown_list = ['CAMEO_DEU_2015','CAMEO_DEUG_2015', 'CAMEO_INTL_2015', 'CAMEO_DEUINTL_2015']
    nine_unknown_list=['KBA05_HERST1', 'KBA05_ALTER4', 'KBA05_SEG3', 'SEMIO_KRIT', 'KBA05_KRSKLEIN', 'KBA05_MOD4', 'KBA05_HERST5', 'KBA05_SEG4', 'KBA05_ZUL3', 'KBA05_ZUL4', 'SEMIO_PFLICHT', 'KBA05_KRSZUL', 'KBA05_FRAU', 'KBA05_KRSAQUOT', 'KBA05_MAXBJ', 'KBA05_ZUL1', 'KBA05_MOTOR', 'KBA05_KW3', 'KBA05_MAXAH', 'KBA05_MOD3', 'SEMIO_SOZ', 'KBA05_MODTEMP', 'KBA05_SEG1', 'KBA05_SEG9', 'SEMIO_FAM', 'SEMIO_KAEM', 'KBA05_ALTER1', 'KBA05_DIESEL', 'KBA05_HERST4', 'KBA05_MAXSEG', 'KBA05_MOTRAD', 'RELAT_AB', 'SEMIO_ERL', 'KBA05_MOD2', 'KBA05_VORB0', 'KBA05_VORB2', 'KBA05_ALTER3', 'KBA05_KW2', 'KBA05_ANHANG', 'KBA05_MOD8', 'KBA05_SEG6', 'KBA05_SEG5', 'KBA05_KRSOBER', 'KBA05_AUTOQUOT', 'KBA05_ALTER2', 'SEMIO_VERT', 'KBA05_KW1', 'KBA05_SEG8', 'KBA05_VORB1', 'ZABEOTYP', 'KBA05_KRSHERST3', 'KBA05_SEG10', 'KBA05_CCM4', 'SEMIO_LUST', 'KBA05_KRSHERST1', 'KBA05_KRSVAN', 'KBA05_MAXHERST', 'KBA05_SEG7', 'SEMIO_RAT', 'KBA05_CCM3', 'KBA05_CCM1', 'KBA05_CCM2', 'KBA05_ZUL2', 'KBA05_MAXVORB', 'KBA05_SEG2', 'KBA05_KRSHERST2', 'SEMIO_KULT', 'KBA05_HERSTTEMP', 'SEMIO_REL', 'SEMIO_DOM', 'SEMIO_TRADV', 'KBA05_HERST3', 'KBA05_HERST2', 'KBA05_MOD1', 'SEMIO_MAT']
    overall_replace_values = [-1, -1.9]
    

    for column in dataset.columns:
        if column in zero_unkown_list:
            dataset[column] = dataset[column].replace([0, -1, -1.9], np.nan)
        elif column in XX_unkown_list:
            dataset[column] = dataset[column].replace(['XX', -1, -1.9, 'X'], np.nan)
        elif column in nine_unknown_list:
            dataset[column] = dataset[column].replace([9, -1, -1.9], np.nan)
        else:
            dataset[column] = dataset[column].replace([-1, -1.9], np.nan)

    if overall_replace_values not in dataset.values:
        print("Successful replacement of incorrectly placed missing values")

    return dataset

def column_information(column):
    print(f"First impression of data {column.name}:")
    print(column.head(5))
    print(f"Column data types: {column.dtypes}")
    print(f"Sum empty values (absolute/percentage): {column.isna().sum()} / {column.isna().sum() / len(column)}")
    values_unique = len(column) == len(column.unique())
    print(f"Rows values unique identifier (Length dataset vs. unique values): {values_unique}")
    if values_unique is False:
        print(column.value_counts())
        column.hist(bins=len(column.unique()))


def removal_NaN_columns(dataset, percentage):
    """
        Calculate the NaN-Percentages per column in a dataset, sort them in descending order and remove columns over a 
        specific threshold

        Args:
            dataset (dataframe): Dataset for NaN percentage calculation
            percentage (float): Percentage threshold by which the respective column will be removed from the dataset

        Returns:
            dataset: Returns dataset with dropped columns (identified through NaN percentage calculation)
    """
    # Calculate the percentages per column and sort them in descending order
    percent_missing = dataset.isnull().sum() * 100 / len(dataset)
    missing_value_df = pd.DataFrame({'column_name': dataset.columns, 'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)

    # select all values larger 25%
    excluded_columns = missing_value_df.loc[missing_value_df['percent_missing'] >= percentage]
    excluded_columns = excluded_columns.index
    print(f"{len(excluded_columns)} columns to be excluded: {excluded_columns}")

    # drop all columns frm dataset 
    dataset = dataset.drop(labels=excluded_columns, axis=1)
    print(f"Remaining columns in dataset: {dataset.shape}")

    return dataset, excluded_columns


def removal_NaN_rows(dataset, percentage):
    """
        Calculate the NaN-Percentages per row in a dataset, and select all rows with a NaN percentage higher than threshold

        Args:
            dataset (dataframe): Dataset for NaN percentage calculation
            percentage (float): Percentage threshold by which the respective row will be removed from the dataset

        Returns:
            dataset: Returns dataset with dropped rows (identified through NaN percentage calculation)
    """
    
    #Convert to numpy array
    np_dataset = dataset.values
    response = dataset['RESPONSE'].values
    
    #Calculate NaN values
    nan_values = np.sum(np.isnan(np_dataset), axis=1) * 100 / len(np_dataset[0])
    nan_row_indices = (nan_values >= percentage).nonzero()[0]
    
    #remove indicies with response = 1
    adj_nan_row_indices = np.array([])
    
    for x in nan_row_indices:
        if response[x] == 0:
            adj_nan_row_indices = np.append(adj_nan_row_indices, x)
    
    #drop rows based on nan_row_indices
    dataset = dataset.drop(adj_nan_row_indices, axis = 0)
    print(f"Remaining shape of dataset: {dataset.shape}")
    
    return dataset


def imputer(df, numerical, binary):
    
    imputer_feature = df.copy()
    
    features_numerical = imputer_feature[numerical]
    features_binary = imputer_feature[binary]
    
    #Impute values with SimpleImputer for binary
    s_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    s_imp = s_imp.fit(features_binary.values)
    features_binary = s_imp.transform(features_binary.values)
    
    #Impute values with KNNImputer for numerical
    KNNimp = KNNImputer()
    KNNimp = KNNimp.fit(features_numerical.values)
    features_numerical = KNNimp.transform(features_numerical.values)
    
    #Add columns and index again
    imputer_feature[binary] = features_binary
    imputer_feature[numerical] = features_numerical
    
    return imputer_feature, s_imp, KNNimp


def imputer_test(df, KNNimp, s_imp, numerical, binary):
    
    imputer_feature = df.copy()
    
    features_numerical = imputer_feature[numerical]
    features_binary = imputer_feature[binary]
    
    #Impute values with SimpleImputer for binary
    features_binary = s_imp.transform(features_binary.values)
    
    #Impute values with KNNImputer for numerical
    features_numerical = KNNimp.transform(features_numerical.values)
    
    #Add columns and index again
    imputer_feature[binary] = features_binary
    imputer_feature[numerical] = features_numerical
    
    return imputer_feature


def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    sum_variance = 0
    components = []
    list_s_values = s.values.squeeze()
    for c in list_s_values:
        components.append(c)
        if len(components) < n_top_components:
            continue
        else:
            sum_variance = sum(list(map(lambda x: x ** 2, components))) / sum(list(map(lambda x: x ** 2, list_s_values)))
            break
    
    return sum_variance


# code to evaluate the endpoint on test data
# returns a variety of model metrics
def evaluate(predictor, test_features, test_labels, prediction, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.  
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """
       
    # LinearLearner produces a `predicted_label` for each data point in a batch
    # get the 'predicted_label' for every point in a batch
    test_preds = prediction
    
    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()
    
    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()
        
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}


def evaluate_model(y_true, y_scores):
    # define evaluation procedure
    score = roc_auc_score(y_true, y_scores)
    return score


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), verbose=None):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring = 'roc_auc',
                       train_sizes=train_sizes, return_times=True, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    print("Roc_auc train score = {}".format(train_scores_mean[-1].round(2)))
    print("Roc_auc validation score = {}".format(test_scores_mean[-1].round(2)))

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# define models to test
# https://machinelearningmastery.com/imbalanced-classification-of-good-and-bad-credit/
def get_models():
    models, names = list(), list()
    # LR
    models.append(LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2'))
    names.append('Logistic Regression')
    # Ada Boost
    names.append('Ada Boost')
    models.append(AdaBoostClassifier())
    # Gradient Boosting
    names.append('Gradient Boosting')
    models.append(GradientBoostingClassifier())
    # RUSBoostClassifier
    names.append('RUSBoost Classifier')
    models.append(RUSBoostClassifier())
    # BalancedRandomForestClassifier
    names.append('RandomForestClassifier')
    models.append(RandomForestClassifier(class_weight='balanced'))
    # BalancedRandomForestClassifier
    names.append('EasyEnsembleClassifier')
    models.append(EasyEnsembleClassifier())
    return models, names


def standard_scaler(df, numerical):
    
    scaled_features = df.copy()
    features = scaled_features[numerical]
    
    #Normalize value with Standard Scaler
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    
    #Add columns and index again
    scaled_features[numerical] = features
    
    return scaled_features, scaler


def standard_scaler_test(df, scaler, numerical):
    
    scaled_features = df.copy()
    features = scaled_features[numerical]
    
    #Normalize value with Standard Scaler
    features = scaler.transform(features.values)
    
    #Add columns and index again
    scaled_features[numerical] = features
    
    return scaled_features
