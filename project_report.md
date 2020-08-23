# Machine Learning Engineer Nanodegree
## Capstone Project
Pelle John
August 15th, 2020

## I. Definition

### Project Overview

The Capstone Project is the final submission for Udacitys Machine Learning Engineer Nanodegree Program. The selected project is based on a real-life example of Arvato, an internationally active services company with a focus on innovations in automation and data/analytics. The requested analysis in about the provided customer data set of a German mail-order sales company to identify customer cluster which can be used to predict the success of targeted marketing campaigns.

The clustering of customers and a respective personalization is a must-have in today's business world. A recent RedPoint Global survey<sup>1</sup> conducted by The Harris Poll that surveyed more than 3,000 consumers in the U.S., U.K., and Canada stated, that

> 63 percent, of consumers expect personalization as a standard of service and believe they are recognized as an individual when sent special offers

It clearly shows that companies need to take into account the characteristics of a customer when choosing the channel, message, and method in a marketing campaign. Multiple studies show, that personalization has a huge impact on the success of a marketing campaign. A study in 2013 by Experian Marketing Services<sup>2</sup> shows that
 
 > personalized promotional emails were shown to lift transaction rates and revenue per email six times higher than non-personalized emails.
 
The relevance in today's business world, as well as the variety of different machine learning methods to be used in this exercise, excites me the most. it gives the possibility to try different methods and explore different libraries and methods while having a clear goal in mind.

The request is split into three main objectives. First, get an overall understanding of the provided customer and German population database to derive an understanding of the data and clean/transform it in a way that it can be used for the following machine learning exercises. 

Secondly, the cleaned data should be used to derive the population cluster which is mapped to the customer database to identify and analyze the relationships between the demographics of the company's existing customers and the general population of Germany. The goal of the second part of the analysis is to describe the parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.

Lastly, the provided understanding should be the basis to train a classification/prediction model based on an existing mailout campaign. In the end, we should be able to use the demographic information from each individual to decide whether or not it will be worth it to include that person in the campaign to and predict the likelihood of response for future mailout campaigns.

The analysis is based on four main datasets, which are all provided by Arvato. The first is a dataset of 891.211 persons (rows) and 366 features (columns) on demographics data for the general population of Germany. It will be used to define a general descriptive cluster for a later mapping of the mail-order company customer base. The customer database consists of  191 652 persons (rows) and 369 features (columns) with demographics data for customers of a mail-order company. It will be used to identify which cluster is in the current customer base of the mail-order company. For classification/prediction model are a train (42 982 persons (rows) with 367 (columns)) and a test dataset (42 982 persons (rows) x 366 (columns)) provided. The train has an additional "RESPONSE" column and will be used to train the classification model for the prediction of success of future targeted marketing campaigns. The test model is without the "RESPONSE" column and can be used in a KAGGLE competition for the success of future targeted marketing campaigns.

### Problem Statement

The problem statement can be split into the main two tasks of the analysis: **Population/Customer Segmentation** and **Classification/Prediction**

#### Population/Customer Segmentation

The key problem to solve is to prepare and cluster the population data in a way, that we can define a clear cluster of interest for the mail-order company, once the customer base is mapped to the trained classificator. Therefore, the key objective is first to transform the dataset, to remove the noise, and to define the major differentiator. 

The data transformation will be done in multiple steps. First, we get an overview of the available German population and customer base dataset (columns, data types, shape, unique, distribution, and percentage of the missing values per column). Once, we have a good understanding of the provided dataset, we start to identify missing values and replace them accordingly. Columns/rows, with a high percentage of missing data, will be removed from the dataset, as they don't provide clear differentiators for the later segmentation. Next, we encode the categorical values to be able to further work with them in the segmentation and transformation process.

After a data normalization step, the simplified German population dataset will be used for a PCA (principal component analysis), as it is a fast and flexible unsupervised method for dimensionality reduction in data. It involves zeroing out one or more of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance<sup>3</sup>.

Once we reduced the noise and dimensions of the segmentation, we will train and optimize a **KMeans cluster algorithm** to detect customer segmentations.
 
The trained cluster model will be used on the customer dataset (which is transformed in the same way as the general population database) to compare the percentages of people per cluster in both datasets. The trained cluster model will show the population cluster, which has a higher/lower percentage as the general Population representation and, therefore, more probably more likely to respond to future marketing/sales campaigns.

#### Classification/Prediction

The key question of the second part of the analysis is: What are effective data transformation/classification approaches to have a stable and accurate prediction on the response rate of individuals on targeted marketing campaigns based on available potential customer demographic data. **The key challenge will be, that we will develop a classification/prediction engine with a very imbalanced dataset (most of the participants did not respond)**. Therefore, some steps in the data preparation will be done slightly differently to compensate for it.

Similar to the first start of the exercise, the analysis will start with the task to understand the provided datasets. After that, we will do similar data cleansing and transformation steps. The data cleansing and transformation will start to replace the missing data, to remove columns with high missing values percentages, to transform categorical values, to remove rows with high missing values percentages, drop non-important columns, replace NaN values and normalize the dataset for a PCA analysis.

The transformed dataset will be split in train and test dataset and then different classification algorithms (SVM, RandomForest,...) and over- and undersampling techniques will be used to train a sufficient classification/prediction model. The trained and evaluated models will be used on the TEST dataset for the KAGGLE competition.

### Metrics

To determine the effectiveness of the used method/model, both parts of the analysis have seperate evalution metrices:

**Population/Customer Segmentation**: J. Kleinberg<sup>5</sup> defined three properties any clustering algorithm should try to satisfy: The axioms of scale invariance, richness, and consistency. He also proved an an impossibility theorem that shows that no clustering algorithm can simultaneously satisfy all of them. In our exercise we willl focus on finding the right number k (number of cluster) via the Elbow method and validated via Silhouette Coefficient, which is used as ground truth labels are not known and the evaluation must be performed using the model itself.

 >The Silhouette Coefficient<sup>6</sup> is an example of such an evaluation, where a higher Silhouette Coefficient score relates to a model with better defined clusters. The Silhouette Coefficient is defined for each sample and is composed of two scores:
 > * **a**: The mean distance between a sample and all other points in the same class.
 > * **b**: The mean distance between a sample and all other points in the next nearest cluster.</br>
 > The Silhouette Coefficient s for a single sample is then given as: <img src="https://render.githubusercontent.com/render/math?math=s = \frac{b - a}{max(a, b)}">

**Classification/Prediction**: For the classification and prediction we will use the accuracy, precision, recall and AUC scores<sup>7</sup>:

* **Accuracy**: Computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions
     <img src="https://render.githubusercontent.com/render/math?math=\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)">```
* **Precision**: Precision is the ability of the classifier not to label as positive a sample that is negative
    ```tp / (tp + fp)```
* **Recall**:  Recall is the ability of the classifier to find all the positive samples
    ```tp / (tp + fn)``` 
* **AUC (Area Under the ROC Curve):** To better evaluate the correctness for the prediction of an inbalanced dataset, the incorporate the AUC mteric, which provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example<sup>8, 9</sup>: <img src="https://render.githubusercontent.com/render/math?math=\frac{2}{c(c-1)}\sum_{j=1}^{c}\sum_{k > j}^c (\text{AUC}(j | k) +\text{AUC}(k | j))">

 
  TP = True Positives; FP = False Positives; FN = False Negatives

## II. Analysis

### Data Exploration

All four datasets for the analysis of  the German mail-order sales company customer have the basis of 366 different columns, which can be split into the following sections:

1. **Personal information (43 columns)**: Information about the respective person in the dataset. This includes, e.g., information on age, gender, and typologies on financial characteristics, life stage, family, social status, and socioeconomic traits.

2. **Household information (32 columns)**: Information about the household the person lives in. This includes, e.g., the number of people, the household structure, transactional activities, and the duration of residence.

3. **Building & postcode and community information (19 columns)**: Information on the building the person lives in. This includes, e.g., the number of households, the type of building, car segments in the neighborhood, the distance from the city center or next metropole, inhabitants, and the share of unemployed persons in the community.

4. **Microcell RR1_ID, RR4_ID & RR3_ID (67 columns)**: Information about a cluster the respective person falls in. This includes the CAMEO typology segmentation and other information like, e.g., the share of car owners in the respective cell, the number of trailers, the number of 1-2 family houses, purchasing power, moving patterns, and online affinity.

5. **AZ Cluster data - 125m x 125m Grid (33 columns)**: Information on transactional data from the mail-order activities for a specific product group for a specific grid. Is based on data from AZ, which has access to 650 Million transaction data.

6. **Postal code related statistics - PLZ8 (114)**: Based on federal German statistics on the postal code. Contains column-like, e.g., the share of car owners per type like BMW, the car engine power, and most-common car types.

**Only for customer dataset**: The customer set has three additional columns "PRODUCT_GROUP, CUSTOMER_GROUP, ONLINE_PURCHASE" which gives information about the respective product group and online_purchase information.

**Only for mailout train dataset**: The dataset has one additional column "REPONSE" to support the training of a supervised classificator.

The datasets for Segmentation/Clustering exercise have the following characteristics:

```
Description about data types in population database:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891221 entries, 0 to 891220
Columns: 366 entries, LNR to ALTERSKATEGORIE_GROB
dtypes: float64(267), int64(93), object(6)
memory usage: 2.4+ GB
None

Description about data types in customer database:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 191652 entries, 0 to 191651
Columns: 369 entries, LNR to ALTERSKATEGORIE_GROB
dtypes: float64(267), int64(94), object(8)
memory usage: 539.5+ MB
None
```

Both datasets have categorical columns and a large set of data. One of the key challenges will be to reduce the number of input data so we can process it efficiently. None of the columns has more than one data type and columns are sorted in alphabetical order. A check for uniqueness of the values revealed, that only one column has more than 500 different data entries and is unique:

````
First impression of data LNR:
0    910215
1    910220
2    910225
3    910226
4    910241
Name: LNR, dtype: int64
Column data types: int64
Sum empty values (absolute/percentage): 0 / 0.0
Rows values unique identifier (Length dataset vs. unique values): True
````
LNR is the index, which we can later use for the KAGGLE competition. A check-in the documentation revealed, that LNR acts as an ID number for each individual in the data partition. The second highest value distribution was in the column ["EINGEFUEGT_AM"]. As this is a piece of plain information on the last update/insert, we also exclude it for further analysis.

**Note:** Some column information are described in more than one column in the dataset (e.g., no. of kids). Also additional columsn exist for the creation of the row, the last update and a unique identifier. Therefore, the number of columns per category do not exactly add up to 366 columns.

### Exploratory Visualization

The number of features is insufficient for a clustering and segmentation algorithm. To better assess the potential, a correlation matrix for the dataset on the German  population was composed to identify potentials for improvement.

![Correlation Matrix](02_images/correlation_matrix.png)

The axes list the features in the German population dataset. The color-encoding represents the correlation between the respective features. Dark blue represents a high correlation close to 1. White represents a low correlation close to 0. A reduced color-density was used to highlight the areas with a very high correlation. 

The correlation matrix shows, that there are multiple feature sets with high correlation. The most important observations are highlighted in red. Especially for the Postal Code related statistics around cars and for the Microcell features. A PCA should be performed to reduce the noise and highlight, extract the most important features in the dataset. 

Secondly, to further identify possible improvement potentials, an analysis was performed to identify the columns with more than 25% missing values. The result was the following:

```
['ALTER_KIND4', 'ALTER_KIND3', 'ALTER_KIND2', 'ALTER_KIND1', 'AGER_TYP', 'EXTSEL992', 'KK_KUNDENTYP', 'ALTER_HH', 'ALTERSKATEGORIE_FEIN', 'D19_LETZTER_KAUF_BRANCHE', 'D19_SOZIALES', 'D19_VERSAND_ONLINE_QUOTE_12', 'D19_LOTTO', 'D19_KONSUMTYP', 'D19_TELKO_ONLINE_QUOTE_12', 'D19_BANKEN_ONLINE_QUOTE_12', 'D19_VERSI_ONLINE_QUOTE_12', 'D19_GESAMT_ONLINE_QUOTE_12']

Number of columns with more than 25% missing values = 18
```

These features are partly redundant based on other features. Therefore, it will be a good step to exclude them from further data cleansing and transformation process. Similar exercises were done for the MAILOUT train dataset for the prediction/clustering algorithm, which resulted in similar results.

### Algorithms and Techniques

The following algorithms/techniques were used for the Segmentation/Clustering and for the Classification/Prediction analysis:

1. [LabelEncoder()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder): Used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.
2. [MinMaxScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html?highlight=minmaxscaler#sklearn.preprocessing.MinMaxScaler): This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one. The transformation is given by:
```python
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```
3. df.fillna(df.mean()): Fill NA/NaN values using the medium of the specific column. Similar to: [SimpleImputer(strategy='mean')](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer).
4. [NaN_replace_KNNImputer(https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer)]: The KNNImputer class provides imputation for filling in missing values using the k-Nearest Neighbors approach. 
5. PCA(): Principal component analysis (PCA) is a technique for reducing the dimensionality of large datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance.<sup>3</sup>

**Population/Customer Segmentation**

6. KMeans()

**Classification/Prediction**

7. SMOTE()
8. RandomUnderSampler()
9. LogisticRegression()
10. GaussianNB()
11. SVM()
12. GaussianProcessClassifier()
13. AdaBoost Classfier()
14. BalancedRandomForestClassifier()
15. GradientBoostingClassifier()
16. RUSBoostClassifier()

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


> The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares This algorithm requires the number of clusters to be specified. It scales well to a large number of samples and has been used across a large range of application areas in many different fields<sup>4</sup>.

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

 <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=0}^{n}\min_{\mu_j \in C}(||x_i - \mu_j||^2)">


### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

-----------

## Sources:
<sup>1</sup> https://www.redpointglobal.com/blog/addressing-the-gaps-in-customer-experience-redpoint-global-harris-poll-benchmark-survey/</br>
<sup>2</sup> https://www.experian.com/assets/marketing-services/white-papers/ccm-email-study-2013.pdf</br>
<sup>3</sup> https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html</br>
<sup>4</sup> https://scikit-learn.org/stable/modules/clustering.html#k-means</br>
<sup>5</sup> https://dl.acm.org/doi/10.5555/2968618.2968676
<sup>6</sup> https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
<sup>7</sup> https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
<sup>8</sup> https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
<sup>9</sup> https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics

