import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def column_information(column):
    print(f"First impression of data {column.name}:")
    print(column.head(5))
    print(f"Column data types: {column.dtypes}")
    print(f"Sum empty values (absolute/percentage): {column.isna().sum()} / {column.isna().sum()/len(column)}")
    values_unique = len(column) == len(column.unique())
    print(f"Rows values unique identifier (Length dataset vs. unique values): {values_unique}")
    if values_unique == False:
        print(column.value_counts())
        hist = column.hist(bins=len(column.unique()))
    pass


def split_dataset(df, size):
    '''
        Split dataset in manageable parts for the respective environment

        Args:
            df (dataframe): The dataset to be split
            size (int): Number of rows you want to split your dataframe into

        Returns:
            list_of_dfs: List of dataframes for further processing
    '''
    
    list_of_dfs = [df.loc[i:i+size-1,:] for i in range(0, len(df),size)]

    return list_of_dfs


def replace_missing_data(dataset):
    '''
        Replace the missing values with NaN-Values 

        Args:
            dataset (dataframe): The dataset to be split
            
        Returns:
            dataset: Processed dataset without wrongly inserted missing values
    '''
    zero_unknown_list = ['ALTER_HH', 'AGER_TYP', 'CJT_GESAMTTYP', 'HEALTH_TYP']
    XX_unknown_list = ['CAMEO_DEU_2015', 'CAMEO_DEUINTL_2015']
    overall_replace_values = [-1, -1.9]
    
    for column in dataset.columns:
        if column in zero_unknown_list:
            dataset[column] = dataset[column].replace([0, -1, -1.9], np.nan)
        elif column in XX_unknown_list:
            dataset[column] = dataset[column].replace(['XX', -1, -1.9], np.nan)
        else:
            dataset[column] = dataset[column].replace([-1, -1.9], np.nan)
    return dataset


def iterator_datasets(list_datasets):
    '''
        Helper function to iterate over all subset of dataset

        Args:
            list_datasets (list): List of dataframes
            
        Returns:
            list_azdias_subsets_edit_missing: List of dataframes withut wrongly classified dataframes
    '''
    list_azdias_subsets_edit_missing = []
    counter = 0
    for i in list_datasets:
        counter += 1
        print(counter)
        subset = replace_missing_data(i)
        list_azdias_subsets_edit_missing.append(subset)
    return list_azdias_subsets_edit_missing


class Dropped_columns:
    def __init__(self):
        self.columns_drop_datasets = []

    def exclude_dropped_column(self, column):
        column_name = column.name
        try:
            self.columns_drop_datasets
        except NameError:
            self.columns_drop_datasets = []
        self.columns_drop_datasets.append(column_name)
        print(self.columns_drop_datasets)
        
    def exclude_dropped_columns(self, column_name):
        try:
            self.columns_drop_datasets
        except NameError:
            self.columns_drop_datasets = []
        self.columns_drop_datasets.extend(column_name.tolist())
        print(self.columns_drop_datasets)


def NaN_percentages(dataset):
    '''
        Calculate the NaN-Percentages per column in a dataset, sort them in descending order

        Args:
            dataset (dataframe): Dataset for NaN percentage calculation
            
        Returns:
            missing_value_df: Returns dataset with percentage NaN values per column
    '''
    percent_missing = dataset.isnull().sum() * 100 / len(dataset)
    missing_value_df = pd.DataFrame({'column_name': dataset.columns, 'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
    return missing_value_df

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
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col].astype(str))
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def minmax_scaler(df):
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df.astype(float)))
    return scaled_df


def NaN_replace_mean(df):
    relaced_mean_df  = df.fillna(df.mean())
    return relaced_mean_df


def iterator_datasets_nomalization(list_datasets):
    '''
        Helper function to iterate over all subset of dataset

        Args:
            list_datasets (list): List of dataframes
            
        Returns:
            list_azdias_subsets_edit_missing: List of dataframes withut wrongly classified dataframes
    '''
    list_azdias_subsets_edit_missing = []
    counter = 0
    for i in list_datasets:
        counter += 1
        print(counter)
        # Fill all NaNs wil mean of respective column
        subset = NaN_replace_mean(i)
        subset_2 = minmax_scaler(subset)
        list_azdias_subsets_edit_missing.append(subset_2)
    return list_azdias_subsets_edit_missing


# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
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


def display_component(v, features_list, component_num, N_COMPONENTS, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()
    

def get_component_list(v, features_list, component_num, N_COMPONENTS, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    return sorted_weight_data
    

# create dimensionality-reduced data
def create_transformed_df(train_pca):
    transformed_dataframe = pd.DataFrame(data=train_pca)
    return transformed_dataframe


# Prepare Customer Data File
def prepare_customer_file(dataset):
    print(dataset.shape)
    
    zero_unknown_list = ['ALTER_HH', 'AGER_TYP', 'CJT_GESAMTTYP', 'HEALTH_TYP']
    XX_unknown_list = ['CAMEO_DEU_2015', 'CAMEO_DEUINTL_2015']
    overall_replace_values = [-1, -1.9]

    #Create separate file with distinct columns
    customer_info = dataset.filter(items = ["PRODUCT_GROUP", "CUSTOMER_GROUP", "ONLINE_PURCHASE"], axis = 'columns')
    dataset = dataset.drop(labels = ["PRODUCT_GROUP", "CUSTOMER_GROUP", "ONLINE_PURCHASE"], axis=1)
    print(customer_info.head(5))
    print(dataset.head(5))
    if "PRODUCT_GROUP" not in dataset.columns:
        print("1. Successful split of unique customer columns!")

    # Replace incorrectly placed NaN values
    list_subsets = hps.split_dataset(dataset, 100000)
    list_customer_edit_missing = hps.iterator_datasets(list_subsets)
    customer_md = pd.concat(list_customer_edit_missing)
    print(customer_md.info())
    if -1 not in customer_md.values:
        print("2. Successful replacement of incorrectly placed missing values")

    customer_td = customer_md.drop(labels=dropped_column.columns_drop_datasets, axis=1)
    print(customer_td .shape)

    # Change the columns wit dtype object to integer values
    customer_td_2 = hps.MultiColumnLabelEncoder(columns = azdias_subset_object).fit_transform(customer_td)
    check_subset_object = customer_td_2.select_dtypes(include=['object']).columns.tolist()

    if not check_subset_object:
        print("3. Object transformation successful, list is empty")
    else:
        print("3. Stop function execution")
        return None

    # Preparation for PCA transformation
    customer_td_2_index = customer_td_2.index
    customer_td_2_columns = customer_td_2.columns

    list_subsets_2 = hps.split_dataset(customer_td_2, 100000)
    list_consumer_td_3 = hps.iterator_datasets_nomalization(list_subsets_2)
    consumer_td_3 = pd.concat(list_consumer_td_3)

    consumer_td_3.index = customer_td_2_index
    consumer_td_3.columns = customer_td_2_columns
    consumer_td_3.head(10)

    if consumer_td_3.values.all() <= 1:
        print("4. Normalization and mean replacement for PCA successful")
        print(consumer_td_3.describe())
    else:
        print("4. Stop function execution")
        return None

    compressed_consumer_td_3 = pca_final.fit_transform(consumer_td_3)
    consumer_td_cp_3 = hps.create_transformed_df(compressed_consumer_td_3)
    consumer_td_cp_3.columns = PCA_column_names

    print(consumer_td_cp_3.head(10))
    print("5. PCA successful")
    
    return consumer_td_cp_3