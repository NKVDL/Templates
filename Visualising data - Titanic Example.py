
# coding: utf-8

# In[127]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:55:46 2018

@author: NVDL
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[128]:


#Importing data
#survivalRate = pd.read_csv('gender_submission.csv')
test_set = pd.read_csv('/Users/NVDL/Code/Contests/Kaggle/Titanic/all/test.csv') 
train_set = pd.read_csv('/Users/NVDL/Code/Contests/Kaggle/Titanic/all/train.csv') 


# In[129]:


train_set.head()


# In[130]:


#Independent Variables
class_ = train_set.iloc[:,2:3].values #Class
sex_ = train_set.iloc[:,4:5].values #Sex
age = train_set.iloc[:,5:6].values #Age
sib = train_set.iloc[:,6:7].values #No. siblings

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(age)
age = imputer.transform(age)
age = age.astype(int)

#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
sex_ = labelencoder.fit_transform(sex_)
#Reshape dataset
sex_ = np.reshape(sex_,(-1,1))


# In[131]:


#Concatenate numpy arrays
X = np.concatenate((age, class_, sex_, sib), axis = 1)
y = train_set.iloc[:,1:2].values #Death


# In[132]:


X


# In[133]:


y

Time for a logistic regression analysis with 4 independent variables: age, sex, class and siblings to predict death. 

Is it possible to derive death from numbers? Well, as it seems this time it can; 4 out of 5 times. You'll see that later.
# In[134]:


#Create test/train set from data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state= 0)

#Feature Scaling x_train and x_test (Z-scores) 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[135]:


X_train


# In[136]:


X_test


# In[137]:


#Create the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train) 


# In[138]:


#Predict y based on x_test
y_pred = classifier.predict(X_test)
y_pred = np.reshape(y_pred,(-1, 1))


# In[139]:


y_pred


# In[140]:


y


# In[141]:


#Making the Confusion Matrix to evaluate the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

114+63/114+63+25+21 = 81% Sensitivity/Accuracy
114/114+21 = 84% Positive Predictive Value
63/63+25 = 72% Specificity 
# In[142]:


# Check missing values in train data
train_set.isnull().sum()


# In[143]:


#Percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' %((train_set['Age'].isnull().sum()/train_set.shape[0])*100))

So, 4 out of 5 times it accurately predicts death on the test set. Under the assumption of prevalence the accuracy goes upward 3%. We now have a classifier that predicts death based on age, sex, no. sibings and class. But what we 'forgot' to check our underlying distrubution of observations of age. We input NaN with the average mean, while the data maybe skewed, because of the missing samples Let's see. 
# In[144]:


ax = train_set["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_set["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

Indeed, the data is skewed. We have to start over. Our classifier was trained under false assumptions. To counter the effect we input not the mean of the observations, but the median. 
# In[145]:


#Mean age
print('The mean of "Age" is %.2f' %(train_set["Age"].mean(skipna=True)))
#Median age
print('The median of "Age" is %.2f' %(train_set["Age"].median(skipna=True)))

Oke. We now know which number to input instead of a missing value. But let's see if we can find more flaws. 
# In[146]:


#Percent of missing "Cabin" 
print('Percent of missing "Cabin" records is %.2f%%' %((train_set['Cabin'].isnull().sum()/train_set.shape[0])*100))


# In[147]:


#Percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' %((train_set['Embarked'].isnull().sum()/train_set.shape[0])*100))


# In[148]:


print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train_set['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_set, palette='Set2')
plt.show()


# In[149]:


print('The most common boarding port of embarkation is %s.' %train_set['Embarked'].value_counts().idxmax())

Now let's fill in the gaps of our missing values in our dataframe all in once.
# In[150]:


train_df = train_set.copy()
train_df["Age"].fillna(train_set["Age"].median(skipna=True), inplace=True)
train_df["Embarked"].fillna(train_set['Embarked'].value_counts().idxmax(), inplace=True)
train_df.drop('Cabin', axis=1, inplace=True)


# In[151]:


#Check missing values in adjusted train dataframe
train_df.isnull().sum()


# In[73]:


#Let's peek at the adjusted train data
train_df.head()


# In[152]:


plt.figure(figsize=(15,8))
ax = train_set["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_set["Age"].plot(kind='density', color='teal')
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_df["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

An eloquent adjustment to properly conduct logistic regression. A smaller SD, lower bias trade off, and smooth conditional probabilities for all inputs of age. Allright, let's move on. Other problems may have occurred that overfitted the classifier. For example, multicollinearity. In the walk of death children are saved first. It might be the case that whoever helped the children, or persons near children had a higher chance of surviving. To simplify, let's concatenate the no. siblings and parch to determine a person's status as 'alone' or 'not alone' = 0,1. And after we're done doing that, let's create dummy variables for every categorical variable and return our dataframe.
# In[153]:


## Create categorical variable for traveling alone
train_df['TravelAlone']=np.where((train_df['SibSp']+train_set["Parch"])>0, 0, 1)
train_df.drop('SibSp', axis=1, inplace=True)
train_df.drop('Parch', axis=1, inplace=True)


# In[158]:


#Create categorical variables and drop some variables
training=pd.get_dummies(train_df, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()


# In[159]:


final_train.isnull().sum()

Nice, let's do the same for our test_dataframe. And we can begin our logistic regression again.
# In[163]:


test_set.isnull().sum()


# In[164]:


test_df = test_set.copy()
test_df["Age"].fillna(train_set["Age"].median(skipna=True), inplace=True)
test_df["Fare"].fillna(train_set["Fare"].median(skipna=True), inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

test_df['TravelAlone']=np.where((test_set["SibSp"]+test_set["Parch"])>0, 0, 1)

test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_df, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()


# In[165]:


final_test.isnull().sum()

Bingo. Let's visualise our dataframes.
# In[166]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

The plot above shows no difference in chances of dying or living with regard to age. So, age does not look like a defining factor for surviving or dying during the titanic event. 
# In[167]:


plt.figure(figsize=(20,8))
avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
plt.show()

#All persons who survived by age. 

There are some peaks at the beginning of the age distribution for survival. Let's add another variable for this effect called 'minor'. 
# In[168]:


final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)

final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)

Let's check out the variable Fare before running our logistic regression.
# In[169]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(-20,200)

plt.show()

Here we see that the chances of dying are highest for those who paid less. Let's see how class associates with the factor of dying. 
# In[170]:


sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
plt.ylim(0,1)
plt.show()

Obviously, a lower class like 3 gives the lowest chance of survival.
# In[171]:


sns.barplot('Embarked', 'Survived', data=train_df, color="lightcoral")
plt.show()

Somehow, persons embarked from Cherbourg, France had the highest chances of survival, Queensland, England came second best chance of survival and lastly Southhampton England. This might have something to do with class. See it this way:

The ticket price of faring on the Titanic from Cherbourg, France was higher and thus, only those with higher class were easier inclined to buy a ticket. 

Let's see how the lone travelers did.
# In[172]:


sns.barplot('TravelAlone', 'Survived', data=final_train, color="mediumturquoise")
plt.show()

As expected (personally) a lower chance of survival in an immediate death-life incident on a sinking prison island for a loner, than for not a loner. Let's see how men and women differ in survival rate.
# In[173]:


sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()

I guess Dicaprio rightfully deserved a higher payroll. The chances of survival based on chromosomal identity is 4 times higher for women than for men. Oke. Let's do our logistic regression again. This time in a different, but somewhat simple format. But wait...

We need to know which features to select though. Let's check by cross validating our features and rank them with recursive elimination to find the optimal set of features. RFE is able to work out the combination of attributes that contribute to the prediction on the target variable (or class). 
# In[177]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X = final_train[cols]
y = final_train['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, 8)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

Now let's create a score for each feature to see which of the following acurcing is proportional to the # of correct classifications.
# In[178]:


from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (# of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

Well, 8 still seems the optimal number for our independent variables. 
# In[181]:


Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                     'Embarked_S', 'Sex_male', 'IsMinor']
X = final_train[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()

Above we see a heatmap, representing the correlation between features of our model for survival. We are ready to run our new logistic regression. 
# In[185]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# create X (features) and y (response)
X = final_train[Selected_features]
y = final_train['Survived']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", and a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))

This plot ROC-curve, of the Power as a function of the Type I Error of the decision rule. The ROC curve is thus the sensitivity as a function of fall-out. In general, if the probability distributions for both detection and false alarm are known, the ROC curve can be generated by plotting the cumulative distribution function (area under the probability distribution from  minus infinity  to the discrimination threshold) of the detection probability in the y-axis versus the cumulative distribution function of the false-alarm probability on the x-axis. Let's evaluate our model now.
# In[184]:


# 10-fold cross-validation logistic regression
logreg = LogisticRegression()
# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
# cv=10 for 10 folds
# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many
scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())


# In[186]:


from sklearn.model_selection import cross_validate

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))

This gives an standard error estimation for our accuracy, log_loss and area under the curve of the ROC curve. But, what would happen to our model now, if we add another variable? Such as 'Fare'. 
# In[188]:


cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
X = final_train[cols]

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, final_train[cols], y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))

Overall, our model drops in performance. Clearly, 'Fare' is bringing noise into the predictability of the our model.Let's use GridSearch to optimize parameter choice for our model.
# In[190]:


from sklearn.model_selection import GridSearchCV

X = final_train[Selected_features]

param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}

gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')

gs.fit(X, y)
results = gs.cv_results_

print('='*20)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, param_grid['C'].max()) 
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()

We can do the same using a pipeline. 
# In[191]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

#Define simple model
###############################################################################
C = np.arange(1e-05, 5.5, 0.1)
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
log_reg = LogisticRegression()

#Simple pre-processing estimators
###############################################################################
std_scale = StandardScaler(with_mean=False, with_std=False)
#std_scale = StandardScaler()

#Defining the CV method: Using the Repeated Stratified K Fold
###############################################################################

n_folds=5
n_repeats=5

rskfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=2)

#Creating simple pipeline and defining the gridsearch
###############################################################################

log_clf_pipe = Pipeline(steps=[('scale',std_scale), ('clf',log_reg)])

log_clf = GridSearchCV(estimator=log_clf_pipe, cv=rskfold,
              scoring=scoring, return_train_score=True,
              param_grid=dict(clf__C=C), refit='Accuracy')

log_clf.fit(X, y)
results = log_clf.cv_results_

print('='*20)
print("best params: " + str(log_clf.best_estimator_))
print("best params: " + str(log_clf.best_params_))
print('best score:', log_clf.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, C.max()) 
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_clf__C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()

Fully prepocessing the data and optimising our hyperparameters with calling GridSearch through a pipeline gives us a maximal accuracy of 80%. Let's fit our pipeline to our dataset and predict for every passenger death. 
# In[200]:


final_test['Survived'] = log_clf.predict(final_test[Selected_features]) #GridSearch classifier
final_test['PassengerId'] = test_df['PassengerId']

submission = final_test[['PassengerId','Survived']]

submission.to_csv("submission.csv", index=False)

submission

That was fun.


