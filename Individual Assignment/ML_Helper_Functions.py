import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from matplotlib.gridspec import GridSpec
from random import randint
import seaborn as sns
sns.set(style="darkgrid")

##############################################################################################################
##############################################################################################################
##############################################################################################################

#Imports CSV file give input path
def read_data(input_path):
    raw_data = pd.read_csv(input_path, keep_default_na=False, na_values=['_'])
    return raw_data

#Plots Scatterplot of Monthly Hours vs Satisfcation (with targets) with boxplots
def plot_descriptive_satisfaction(data):
    fig = plt.figure(figsize=(9, 4))
    gs = GridSpec(1, 3, width_ratios=[3, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0 = plt.scatter(data.average_montly_hours[data.left==0], 
                      data.satisfaction_level[data.left==0], 
                      label='left=No',
                      marker='.', c='red', alpha=0.5)
    ax0 = plt.scatter(data.average_montly_hours[data.left==1], 
                      data.satisfaction_level[data.left==1],
                      label='left=Yes',
                      marker='+', c='green', alpha=0.7)
    ax0 = plt.xlabel('average_montly_hours')
    ax0 = plt.ylabel('satisfaction_level')
    ax0 = plt.legend(loc='best')
    ax0 = plt.subplot(gs[1])
    ax1 = sns.boxplot(x="left", y="average_montly_hours", data=data)
    ax0 = plt.subplot(gs[2])
    ax2 = sns.boxplot(x="left", y="satisfaction_level", data=data)

    plt.tight_layout()
    plt.show()

#Plots Scatterplot of Monthly Hours vs Last Evaluation (with targets) with boxplots
def plot_descriptive_last_evaluation(data):
    fig = plt.figure(figsize=(9, 4))
    gs = GridSpec(1, 3, width_ratios=[3, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0 = plt.scatter(data.average_montly_hours[data.left==0], 
                      data.last_evaluation[data.left==0], 
                      label='left=No',
                      marker='.', c='red', alpha=0.5)
    ax0 = plt.scatter(data.average_montly_hours[data.left==1], 
                      data.last_evaluation[data.left==1],
                      label='left=Yes',
                      marker='+', c='green', alpha=0.7)
    ax0 = plt.xlabel('average_montly_hours')
    ax0 = plt.ylabel('last_evaluation')
    ax0 = plt.legend(loc='best')
    ax0 = plt.subplot(gs[1])
    ax1 = sns.boxplot(x="left", y="average_montly_hours", data=data)
    ax0 = plt.subplot(gs[2])
    ax2 = sns.boxplot(x="left", y="last_evaluation", data=data)

    plt.tight_layout()
    plt.show()

#Returns numerical variables
def numerical_features(df):
    columns = df.columns
    return df._get_numeric_data().columns

#Returns categorical variables
def categorical_features(df):
    numerical_columns = numerical_features(df)
    return(list(set(df.columns) - set(numerical_columns)))

#Returns booleans variables
def boolean_features(df):
    boolean_columns = numerical_features(df)
    boolean_columns = [x for x in boolean_columns if (len(df[x].unique()) == 2) 
                        and (0 in df[x].unique()) and (1 in df[x].unique())]
    return boolean_columns

#Plots distribution of given categorical variable
def categorical_plot(df, categorical_features, var):
    plt.figure(figsize=(8,5)) 
    sns.countplot(df[categorical_features[var]])
    plt.show()

#First create scale variable for salary variable
def categorical_to_scale(df, var):
    new_df = df.copy()
    unique_val = np.unique(df[var])

    new_df['sal_band'] = [2 if df[var][x] == unique_val[0] else 0 if df[var][x] == 
                          unique_val[1] else 1 for x in range(0,len(df[var]))]
    
    return new_df

#Encodes categorical variables
def onehot_encode(df, override = []):
    numericals = df.get(numerical_features(df))
    new_df = numericals.copy()

    if len(override) >= 1:
        cat = override
    else:
        cat = categorical_features(df)
    for categorical_column in cat:
        new_df = pd.concat([new_df, pd.get_dummies(df[categorical_column], prefix = 
                categorical_column)], axis=1)
        
    return new_df

#Gets the distribution of given list of booleans
def boolean_dist(df, bools):
    output = []
    for col in bools:
        dist = df[col].value_counts()
        output.append(dist)
    return output

#Small multiples of numerical value histograms
def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()
    plt.show()

#Checks which features have skewness present
def feature_skewness(df):
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    numeric_features = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes: 
            numeric_features.append(i)

    feature_skew = df[numeric_features].apply(
        lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':feature_skew})
    return feature_skew, numeric_features

def fix_skewness(df):
    feature_skew, numeric_features = feature_skewness(df)
    high_skew = feature_skew[feature_skew > 0.5]
    skew_index = high_skew.index
    
    #Create copy
    df = df.copy()
    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

    skew_features = df[numeric_features].apply(
        lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':skew_features})
    return df

def standardize(df, numerical_values):
    standardized_numericals = preprocessing.scale(df[numerical_values])
    df[numerical_values] = standardized_numericals
    
    return df

#Remove highly correlated variables
def correlation_removal(y_set, percent = 0.99):
    corr_matrix = y_set.corr().abs()

    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    #Find index of feature columns with correlation greater than given percentage
    to_drop = [column for column in upper.columns if any(upper[column] >= percent)]

    print('Need to remove {} columns with >= 0.99 correlation.'.format(len(to_drop)))
    y_set = y_set[[x for x in y_set if x not in to_drop]]
    
    return y_set

#Plots heatmap of confusion matrix
def confusion_heat_map(test_set, prediction_set):
    cm = confusion_matrix(test_set, prediction_set)

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def score_model(data, dependent_var, size, seed):
    X = data.loc[:, data.columns != dependent_var]
    y = data.loc[:, dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    
    # Create logistic regression object
    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(X_train, y_train)   
    return classifier.score(X_test, y_test)

def cv_evaluate(df, target_var, seed, C_input = 1000, max_input = 100):
    # Create logistic regression object
    lm = LogisticRegression(solver='lbfgs', C = C_input, max_iter = max_input)
    kfolds = KFold(n_splits=10, shuffle=True, random_state=seed)

    X = df.drop([target_var], axis=1)
    y = df.left.reset_index(drop=True)
    benchmark_model = make_pipeline(RobustScaler(), lm).fit(X=X, y=y)
    scores = cross_val_score(benchmark_model, X, y, scoring='accuracy', cv=kfolds)   
    return scores[scores >= 0.0]

def ROC_curve(model, y_test_set, X_test_set):
    logit_roc_auc = roc_auc_score(y_test_set, model.predict(X_test_set))
    fpr, tpr, thresholds = roc_curve(y_test_set, model.predict_proba(X_test_set)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

#Remove outliers from exogenous variables
def remove_outliers(df):
    X = df.drop(['left'], axis=1)
    y = df.left.reset_index(drop=True)
    ols = sm.OLS(endog = y, exog = X)
    fit = ols.fit()
    test = fit.outlier_test()['bonf(p)']
    outliers = list(test[test<1e-3].index) 
    df.drop(df.index[outliers])
    return df

#Calculates average projects per year
def projects_per_year(df):
    df['projects_per_year'] = df['number_project']/df['time_spend_company']    
    return df

#Removes under represented features
def under_represented_features(df):
    under_rep = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if ((zeros / len(df)) * 100) > 99.0:
            under_rep.append(i)
    df.drop(under_rep, axis=1, inplace=True)
    return df

#Create binned variable for a given continuous variable
def binning(col, cut_points, labels=None):
    #Define min and max values:
    min_val = col.min()
    max_val = col.max()    
    #create list by adding min and max to cut_points
    break_points = [min_val] + cut_points + [max_val]
    #if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)
    #Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin

#Choose variable to be binned and passed to binning function, output updated df
def bin_continuous_var(df, override = ''):
    numerical = numerical_features(df)
    booleans = boolean_features(df)
    
    #Choose only continuous numerical variables to be binned
    binned_variables = (list(set(numerical) - set(booleans)))
    
    if len(override) > 0:
        binned_variables.remove(override)
    
    #Bin based on median and 1 standard deviation above and below
    for var in binned_variables:
        var_nam = 'binned_' + str(var)
        cut_points = [df[var].median(axis = 0) - df[var].std(axis = 0), df[var].median(axis = 0), 
                      df[var].median(axis = 0) + df[var].std(axis = 0)]
        df[var_nam] = binning(df[var], cut_points)
    
    return df

#Iteratively cycles throughout input feature engineering functions and determines their effect on the model
def feature_engineering_pipeline(raw_data_total, raw_data_test, dependent_var, sample_size, seed, fe_functions):
    selected_functions = []
    base_score = score_model(raw_data_test, dependent_var, sample_size, seed)
    print('Base Accuracy on Training Set: {:.4f}'.format(base_score))
    #Applying approved engineering on entire dataset, but testing its validity only on test set
    engineered_data_total = raw_data_total.copy()
    engineered_data_test = raw_data_test.copy()
    for fe_function in fe_functions:
        processed_data_total = globals()[fe_function](engineered_data_total)
        processed_data_test = globals()[fe_function](engineered_data_test)
        new_score = score_model(processed_data_test, dependent_var, sample_size, seed)
        print('- New Accuracy ({}): {:.4f} '.format(fe_function, new_score), 
              end='')
        difference = (new_score-base_score)
        print('[diff: {:.4f}] '.format(difference), end='')
        if difference > -0.01:
            selected_functions.append(fe_function)
            engineered_data_total = processed_data_total.copy()
            engineered_data_test = processed_data_test.copy()
            base_score = new_score
            print('[Accepted]')
        else:
            print('[Rejected]')
    return selected_functions, engineered_data_total

def feature_reduction(model, score_target, X_entire_set, X_train_set, y_train_set):
    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(model, step=1, cv=10, scoring=score_target)
    rfecv.fit(X_train_set, y_train_set.values.ravel())

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_train_set.columns[rfecv.support_]))

    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(8,5))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return X_entire_set[X_entire_set.columns[rfecv.support_]], rfecv.estimator_

#Remove engineered features that resulted in invalid entries
def NaN_removal(df):
    drop_cols = [x for x in df if df[x].isnull().sum() == len(df)]
    output_df = df.drop(drop_cols, axis = 1)
    print('Need to remove {} columns with invalid entries.'.format(len(drop_cols)))
    return output_df

#Standardization for Polynomial Feature
def standardize2(df):
    standardized_numericals = preprocessing.scale(df)
    df = standardized_numericals  
    return df

#Returns list of prediction accuracies
def get_accuracy_list(model,X_test,y_test,y_pred):
    pred_proba_df = pd.DataFrame(model.predict_proba(X_test))
    threshold_list = np.arange(0.05, 1.0, 0.05)
    accuracy_list = np.array([])
    for threshold in threshold_list:
        y_pred = pred_proba_df.applymap(lambda prob: 1 if prob > threshold else 0)
        test_accuracy = accuracy_score(y_test.values,
                                    y_pred[1].values.reshape(-1, 1))
        accuracy_list = np.append(accuracy_list, test_accuracy)
    return accuracy_list, threshold_list

#Plots chart of accuracies
def accuracy_plot(accuracy_list, threshold_list):
    plt.plot(range(accuracy_list.shape[0]), accuracy_list, 'o-', label='Accuracy')
    plt.title('Accuracy for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.xticks([i for i in range(1, accuracy_list.shape[0], 2)], np.round(threshold_list[1::2], 1))
    plt.grid()
    plt.show()
