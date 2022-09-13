#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:tomato;text-align:center;font-size:300%;font-family:verdana;"> Cancer Classification Assignment</h1>

# ___
# For this assignment we'll try to predict a cancer class - benign or malignant.
# We'll use:<br>
# A. Decision Tree<br>
# B. Random Forest<br>
# C. Adaboost
# 
# <b>NOTE: all visualizations must have axes names and title and large enough to read</b>
# ____

# <div class="alert alert-info"><strong>NOTE:</strong> Remember the steps:<br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>1. <b><u>Imports</u> -></b> Import relevant libraries and data sets</tt></font><br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>2. <b><u>Exploration</u> -></b> Inspecting the data and exploration data analysis</tt></font><br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>3. <b><u>Manipulation</u> -></b> Data alteration, adding/removing columns, etc. <i><b><span style="background-color:yellow">OPTIONAL</span></b></i></tt></font><br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>4. <b><u>Algorithm</u> -></b> Splitting the data to train set and test set, training the algorithm. </tt></font><br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>5. <b><u>Validation</u> -></b> Checking metric accuracies on the test set.</tt></font><br>
#     &nbsp;&nbsp;&nbsp;&nbsp;<font color=black><tt>6. <b><u>Conclusion</u> -></b> Conclusion on performance, model selection, business problem.</tt></font></div>

# ___
# # <u>Imports</u>
# You may need to state %matplotlib inline, depending on your version of matplotlib

# #### Import Libraries

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import warnings 
warnings.filterwarnings("ignore")


# #### Import the data (cancer_classification.csv) and print the head of the data

# In[2]:


data = pd.read_csv('cancer_classification.csv')
datacopy = pd.read_csv('cancer_classification.csv')
pd.set_option('max_columns',None)
data.head(10)


# ____
# # <u>Exploration</u>

# #### Check the features data types

# In[3]:


data.info()


# #### Generate descriptive statistics and [transpose](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html) the data frame

# In[4]:


data.transpose()
data.describe()


# #### Check for null values

# In[5]:


print(data.isnull().sum())
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# ### <u>Exploratory Data Analysis - visualization</u>

# #### How many people have cancer? - countplot

# In[6]:


plt.figure(figsize=(8,6))
sns.countplot(data['benign_0__mal_1']);
plt.title("How many people have cancer (0=benign , 1=malignant)?",fontsize=15);


# #### Use the mean area against the mean smoothness to see how many people have cancer - scatterplot

# In[7]:


plt.figure(figsize=(8,6))
sns.scatterplot(data['mean area'],data['mean smoothness']);


# #### Use the mean smoothness against the mean texture to see how many people have cancer - scatterplot

# In[8]:


plt.figure(figsize=(8,6))
sns.scatterplot(data['mean smoothness'],data['mean texture']);


# #### Use the mean texture against the mean symetry to see how many people have cancer - scatterplot

# In[9]:


plt.figure(figsize=(8,6))
sns.scatterplot(data['mean texture'],data['mean symmetry']);


# #### Use the worst fractal dimension against the mean texture to see how many people have cancer - scatterplot

# In[10]:


plt.figure(figsize=(8,6))
sns.scatterplot(data['worst fractal dimension'],data['mean texture']);


# #### Use heatmap to show the [correlation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) between the features 
# (use cmap='viridis' and print the annotations)

# In[11]:


sns.heatmap(data.corr(),yticklabels=False,cbar=False,cmap='viridis');


# #### Use heatmap to show the correlation between the mean features - all the features that have the word 'mean'
# (use cmap='viridis' and print the annotations)<br>
# hint: you need to save them first to a new variable

# In[12]:


data_mean = data[['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension']]
sns.heatmap(data_mean.corr(),yticklabels=False,cbar=False,cmap='viridis');


# ____
# # <u>Algorithm</u>

# #### Split the data to X and y

# In[13]:


X = data.drop('benign_0__mal_1', axis=1)
y = data['benign_0__mal_1']


# #### Split the data to train set and test set (30% test set, random state = student's ID number)

# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# #### Scale the data
# Remember: we don't want to suffer from data leakage so we fit and transform the training set, while only transforming the test set

# In[15]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ___
# ## <u>Decision Tree - Gini</u>

# In[16]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import Image  
import pydotplus
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve


# #### Instatiate the algorithm (max_depth=3, min_sample_leaf=2, random_state = ID number)

# In[17]:


gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=2, random_state=101)


# #### Fit the training set

# In[18]:


gini.fit(X_train, y_train)


# #### Print the tree with graphviz

# In[19]:


target_names = ['0', '1'] 
dot_data = tree.export_graphviz(gini, out_file=None, 
                                feature_names=list(X.columns),  
                                class_names=target_names, filled=True ) #creating the tree

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())


# ## <u>Decision Tree - Gini - Accuracy Metrics</u>

# #### Print the accuracy and the roc auc

# In[20]:


y_pred = gini.predict(X_test)

print('accuracy:','',(accuracy_score(y_test,y_pred)*100).round(2),'%')
print('roc auc:','',(roc_auc_score(y_test,y_pred)*100).round(2),'%')


# #### Print the Confusion Matrix and Classification Report

# In[21]:


print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))


# #### Print the ROC curve

# In[22]:


probs = gini.predict_proba(X_test) 
preds = probs[:,1:] 
fpr, tpr, thresholds = roc_curve(y_test, preds)
auc = roc_auc_score(y_test, preds) 
plt.figure(figsize=(8,6)) 
plt.plot(fpr, tpr, linewidth=2,label="Gini Decision Tree , auc="+str(auc))
plt.plot([0,1], [0,1], 'k--' ) 
plt.rcParams['font.size'] = 12 
plt.title('ROC curve for Gini Decision Tree Classifier for Predicting Cancer') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.grid(b=True)
plt.show()


# ___
# ## <u>Decision Tree - Entropy</u>

# #### Repeat the steps for entropty as well including accuracy

# In[23]:


entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=2, random_state=101)

entropy.fit(X_train, y_train)


# In[24]:


entropy.fit(X_train, y_train)


# In[25]:


dot_data = tree.export_graphviz(entropy, out_file=None, 
                                feature_names=list(X.columns),  
                                class_names=target_names, filled=True )

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())


# In[26]:


y_pred = entropy.predict(X_test)

print('accuracy:','',(accuracy_score(y_test,y_pred)*100).round(2),'%')
print('roc auc:','',(roc_auc_score(y_test,y_pred)*100).round(2),'%')


# In[27]:


print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))


# In[28]:


probs = entropy.predict_proba(X_test)
preds = probs[:,1:]
fpr, tpr, thresholds = roc_curve(y_test, preds)
auc = roc_auc_score(y_test, preds)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2,label="Entropy Decision Tree , auc="+str(auc))
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Entropy Decision Tree Classifier for Predicting Cancer')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.grid(b=True)
plt.show()


# ___
# ## <u>Random Forest</u>

# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# #### Use grid search to fit the best parameters - might take a while depending on your CPU

# In[30]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators':[50, 100, 400, 700, 1000], 
             'max_depth':[2, 4, 10, 12, 16], 
             'criterion':['gini','entropy']} 


# In[31]:


grid = GridSearchCV(RandomForestClassifier(),param_grid,verbose=3)
grid.fit(X_train,y_train)


# #### What are the grid's [best parameters](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)?

# In[32]:


grid.best_params_


# #### Print the accuracy and roc auc

# In[33]:


grid_pred = grid.predict(X_test)
print('accuracy:','',(accuracy_score(y_test,grid_pred)*100).round(2),'%')
print('roc auc:','',(roc_auc_score(y_test,grid_pred)*100).round(2),'%')


# #### Print the Confusion Matrix and Classification Report

# In[34]:


print(confusion_matrix(y_test,grid_pred))
print()
print(classification_report(y_test,grid_pred))


# #### Print the ROC curve

# In[35]:


probs = grid.predict_proba(X_test)
preds = probs[:,1:]
fpr, tpr, thresholds = roc_curve(y_test, preds)
auc = roc_auc_score(y_test, preds)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2,label="Random Forest , auc="+str(auc))
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Random Forest Classifier for Predicting Cancer')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.figtext(1, 0.5, "Criterion': 'gini', \nMax Depth: 10, \nNumber of Estimators: 50", size='large')
plt.legend(loc=4)
plt.grid(b=True)
plt.show()


# #### Optional - print the feature importance of the selected model

# In[36]:


rfc=RandomForestClassifier(n_estimators = 50, max_depth = 10, criterion='gini',random_state=101)
rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)


# In[37]:


pd.concat((pd.DataFrame(X.columns, columns = ['variable']), 
           pd.DataFrame(rfc.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# ___
# ## <u>Adaboost</u>

# In[38]:


from sklearn.ensemble import AdaBoostClassifier


# #### Instantiate the model with 200 estimators and repeat all the steps of fit, predict, accuracy matrics and feature importance (optional)

# In[39]:


clf = AdaBoostClassifier(n_estimators=200, random_state=101)


# In[40]:


clf.fit(X_train, y_train)


# In[41]:


y_pred = clf.predict(X_test)


# In[42]:


print('accuracy:','',(accuracy_score(y_test,y_pred)*100).round(2),'%')
print('roc auc:','',(roc_auc_score(y_test,y_pred)*100).round(2),'%')


# In[43]:


print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))


# In[44]:


probs = clf.predict_proba(X_test)
preds = probs[:,1:]
fpr, tpr, thresholds = roc_curve(y_test, preds)
auc = roc_auc_score(y_test, preds)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2,label="Adaboost , auc="+str(auc))
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Adaboost Classifier for Predicting Cancer')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.grid(b=True)
plt.show()


# In[45]:


pd.concat((pd.DataFrame(X.columns, columns = ['variable']), 
           pd.DataFrame(clf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# ___
# ## <u>Conclusion</u>

# #### What is the best algorithm? (Decision Tree Gini, Decision Tree Entropy, Random Forest, Adaboost) - answer in details

# The best algorithm is AdaBoost because when we look at the results of the accurasy and roc auc of every algorithm we can clearly see that the results of the Adaboost are the best.
# 
# The results for the Adaboost are: (accuracy:  96.49 % ,  roc auc:  96.02 %)
# 
# Unlike the results for the other algorithms: 
# Gini: (accuracy:  94.15 % , roc auc:  93.55 %) , Entropy: (accuracy:  accuracy:  93.57 % , roc auc:  92.79 %), RandomForest: (accuracy:  95.91 % , roc auc:  95.54 %)
# 
# We can also see the difference between the ROC curve of every algorithm, the Adaboost algorithm is the best because it is the closest to the left top edge
