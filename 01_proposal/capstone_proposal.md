
# Machine Learning Engineer Nanodegree
## Capstone Proposal
Pelle John</br>
August 15th, 2020

## Proposal: *Customer segmentation and prediction using Machine Learning models for a targeted marketing campaign*

### Domain Background

Arvato, as an internationally active services company with a focus on innovations in automation and data/analytics was asked by a mail-order sales company to identify customer cluster which can be used to predict the success of targeted marketing campaigns.

The clustering of customers and a respective personalization is a must-have in today's business world. A recent RedPoint Global survey<sup>1</sup> conducted by The Harris Poll that surveyed more than 3,000 consumers in the U.S., U.K., and Canada stated, that

> 63 percent, of consumers expect personalization as a standard of service and believe they are recognized as an individual when sent special offers

 It clearly shows that companies need to take into account the characteristics of a customer when choosing the channel, message, and method in a marketing campaign. Multiple studies show, that personalization has a 
 huge impact on the success of a marketing campaign. A study in 2013 by Experian Marketing Services<sup>2</sup> shows that
 
 > personalized promotional emails were shown to lift transaction rates and revenue per email six times higher than non-personalized emails.
 
 The relevance in today's business world, as well as the variety of different machine learning methods to be used in this exercise, excites me the most. it gives the possibility to try different methods and explore different libraries and methods while having a clear goal in mind. 
 
### Problem Statement

The key question of the exercise is: What are the effective cluster to reach a high-conversion/response rate for targeted marketing campaigns based on current population data and respective current customer base data?

The project is about the analysis of demographics data for a mail-order sales company in Germany. The exercise starts with the correct identification of demographic clusters of the general population of Germany. Once the cluster is identified,  the customer base is mapped to the identified segments to determine the characteristics and demographic information of the customer base. The customer segments are the basis for a classification model of a past marketing campaign and then applied on a third dataset for targets of a marketing campaign of the company to determine the prediction in which individuals are most likely to convert into becoming customers for the company.

### Datasets and Inputs

The datasets used for the analysis are all provided by Arvato. There are four data files and two meta data files associated with this project:

1. **Udacity_AZDIAS_052018.csv**: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns). The general population will be used to define a general descriptive cluster for a later mapping of the mail-order company customer base
1. **Udacity_CUSTOMERS_052018.csv**: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns). Will be used to identify which cluster is in the current customer base of the mail-order company. The clustering of the customer base will be used to define a classification/prediction model for future targeted marketing campaigns 
1. **Udacity_MAILOUT_052018_TRAIN.csv**: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns). Will be used to train the classification model for the prediction of success of future targeted marketing campaigns
1. **Udacity_MAILOUT_052018_TEST.csv**: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns). Will be used to test the classification model for the prediction of success of future targeted marketing campaigns
1. **Meta: DIAS Information Levels - Attributes 2017.xlsx**: Top-level list of attributes and descriptions, organized by informational category. Will be used to interpret the columns in the customer dataset and the general population dataset.
1. **Meta: DIAS Attributes - Values 2017**: Detailed mapping of data values for each feature in alphabetical order. Will be used to support the data cleansing and transformation process prior to the segmentation, classification and analysis.

### Solution Statement

The Bertelsmann/Arvato capstone project can be split into three main exercises. The first part is on the <b>investigation and cleansing of a large population dataset</b>, which is then used in the second step for an <b>unsupervised customer segmentation model</b>. In the final step, the clusters are used in combination with the existing customer base to develop a <b>supervised classification model</b>. The model is then used to predict the success of a targeted customer base. The tasks can be solved with the following machine learning methods:</br>

1.) Data cleansing, transformation and dimensionality reduction through <b>randomized PCA</b></br>
> Principal component analysis (PCA) is a technique for reducing the dimensionality of large datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance.<sup>3</sup>

2.) Clustering models to segment customers and German population dataset with a focus on <b>MiniBatch KMeans</b></br>
> The MiniBatchKMeans is a variant of the KMeans algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function. Mini-batches are subsets of the input data, randomly sampled in each training iteration. These mini-batches drastically reduce the amount of computation required to converge to a local solution. In contrast to other algorithms that reduce the convergence time of k-means, mini-batch k-means produces results that are generally only slightly worse than the standard algorithm.<sup>4</sup>

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

 <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=0}^{n}\min_{\mu_j \in C}(||x_i - \mu_j||^2)">

3.) Classification/regression models to predict the success of different methods for the targeted customer base with a focus on <b>SGD Classifier</b>
> Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Stochastic gradient descent is an optimization method for unconstrained optimization problems. In contrast to (batch) gradient descent, SGD approximates the true gradient of  <img src="https://render.githubusercontent.com/render/math?math=E(w,b)"> by considering a single training example at a time. The class SGDClassifier implements a first-order SGD learning routine. The algorithm iterates over the training examples and for each example updates the model parameters according to the update rule given by<sup>5</sup> 

 <img src="https://render.githubusercontent.com/render/math?math=w \leftarrow w - \eta \left[\alpha \frac{\partial R(w)}{\partial w} + \frac{\partial L(w^T x_i + b, y_i)}{\partial w}\right]">
 
Note: The classification/regression model could be changed after initial testing to enhance the performance of the targeting marketing campaign predictor. Other classifiers, which could be interesting, are Decision Tree Classifiers/Random Forests or SVMs.

### Benchmark Model

A benchmark is best suited for the classification exercise to predict the success of a targeting marketing campaign. Similar exercises on Kaggle (see below) assume a prediction rate of 80% to 90% accuracy. That is what we will aim for in this exercise.

Similar Kaggle Notebooks/ Competitions:
* https://www.kaggle.com/c/springleaf-marketing-response
* https://www.kaggle.com/janiobachmann/bank-marketing-campaign-opening-a-term-deposit
* https://www.kaggle.com/c/marketing-campaign-effectiveness/overview

### Evaluation Metrics

For each of the three steps there are evaluation metrics which will be using to determine the effectiveness of the used method/model.

1. **Principal Component Analysis**: The most common evaluation metrics are CPV (cumulative percent variance), which describes the accounted accumulated variance for the designed and selected features. While there are other metrics (Parallel Analysis, Cross-validation) which could be used as evaluation metrics, we will stick with the CPV as the most popular one.
2. **Clustering**: J. Kleinberg<sup>6</sup> defined three properties any clustering algorithm should try to satisfy: The axioms of scale invariance, richness, and
consistency. He also proved an an impossibility theorem that shows that no clustering algorithm can simultaneously satisfy all of them. In our exercise we willl focus on finding the right number k (number of cluster) via the Elbow method and validated via Silhouette Coefficient, which is used as ground truth labels are not known and the evaluation must be performed using the model itself.

 >The Silhouette Coefficient<sup>7</sup> is an example of such an evaluation, where a higher Silhouette Coefficient score relates to a model with better defined clusters. The Silhouette Coefficient is defined for each sample and is composed of two scores:
 > * **a**: The mean distance between a sample and all other points in the same class.
 > * **b**: The mean distance between a sample and all other points in the next nearest cluster.</br>
 > The Silhouette Coefficient s for a single sample is then given as: <img src="https://render.githubusercontent.com/render/math?math=s = \frac{b - a}{max(a, b)}">

3. **Classification/Prediction**: For the classification and prediction we will use the Accuracy, precision and recall scores<sup>8</sup>:

    * **Accuracy**: Computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions
     <img src="https://render.githubusercontent.com/render/math?math=\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)">```
    * **Precision**: Precision is the ability of the classifier not to label as positive a sample that is negative
    ```tp / (tp + fp)```
    * **Recall**:  recall is the ability of the classifier to find all the positive samples
    ```tp / (tp + fn)``` 
 
  TP = True Positives; FP = False Positives; FN = False Negatives
 
### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

## Sources:
<sup>1</sup> https://www.redpointglobal.com/blog/addressing-the-gaps-in-customer-experience-redpoint-global-harris-poll-benchmark-survey/</br>
<sup>2</sup> https://www.experian.com/assets/marketing-services/white-papers/ccm-email-study-2013.pdf</br>
<sup>3</sup> https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202#:~:text=Principal%20component%20analysis%20(PCA)%20is,variables%20that%20successively%20maximize%20variance</br>
<sup>4</sup> https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans</br>
<sup>5</sup> https://scikit-learn.org/stable/modules/sgd.html#sgd</br>
<sup>6</sup> https://dl.acm.org/doi/10.5555/2968618.2968676
<sup>7</sup> https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
<sup>8</sup> https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

**Before submitting your proposal, ask yourself. . .**

- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
