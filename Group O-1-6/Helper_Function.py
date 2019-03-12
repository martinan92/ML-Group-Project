import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from matplotlib.gridspec import GridSpec
from gplearn.genetic import SymbolicTransformer
from random import randint
from math import sin, cos, sqrt, asin, radians
import seaborn as sns
sns.set(style="darkgrid")

##############################################################################################################
##############################################################################################################
##############################################################################################################

#Imports CSV file give input path
def read_data(input_path):
    raw_data = pd.read_csv(input_path, keep_default_na=False, na_values=['_'])
    return raw_data

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
    boolean_columns = df.columns
    boolean_columns = [x for x in boolean_columns if ((len(df[x].unique()) == 2) and (0 in df[x].unique() and 
                      1 in df[x].unique())) or ('FALSE' in df[x].unique() and 'TRUE' in df[x].unique())]
    return boolean_columns

#Checks given dataframe for nulls and returns columns where there are any present
def null_check(df):
    if df.isnull().values.any() == False and df.isna().values.any() == False:
        print("No nulls present.")
    else:
        null_count = df.columns[df.isnull().any()].tolist()
        na_count = df.columns[df.isna().any()].tolist()
        blank_count = null_count.append(na_count)
        print("The following columns have nulls: ", blank_count)

#Plots density plot against target variable for given scale variables
def density_plot(df, var, lower_bound, upper_bound):
    plt.figure(figsize=(15,8))
    ax = sns.kdeplot(df[var][df.status_group == 'functional'], 
                    color="forestgreen", shade=True)
    sns.kdeplot(df[var][df.status_group == 'non functional'], 
                color="lightcoral", shade=True)
    sns.kdeplot(df[var][df.status_group == 'functional needs repair'], 
                color="darkturquoise", shade=True)
    plt.legend(['Functional', 'Non-Functional', 'Functional Needs Repair'])
    plt.title('Density Plot of {}'.format(var))
    ax.set(xlabel=var)
    plt.xlim(lower_bound,upper_bound)
    plt.show()
    
#Plots distribution of given categorical variable
def categorical_plot(df, categorical_features, var):
    plt.figure(figsize=(15,7)) 
    sns.countplot(df[categorical_features[var]])
    plt.show()

#Stacked bar chart for a categorical variable against the target
def cat_stacked_bar(df, target, var):
    plt.figure(figsize=(15,7)) 
    df_plot = df.groupby([target, var]).size().reset_index().pivot(columns=target, index=var, values=0)
    df_plot.plot(kind = 'bar', stacked = True)
    plt.show()

#Remove variables from category list that are too large or too small
def drop_categorical(df, categoricals, upper_bound, lower_bound):
    reduced_cat = categoricals.copy() 
    large_drop = [column for column in reduced_cat if (df[column].nunique() > upper_bound)]
    small_drop = [column for column in reduced_cat if (df[column].nunique() < lower_bound)]
    reduced_cat = [x for x in reduced_cat if x not in large_drop]
    reduced_cat = [x for x in reduced_cat if x not in small_drop]

    print('The following categories have too many unique values:', large_drop)
    print('The following categories have too few unique values:', small_drop)
    
    return reduced_cat, large_drop, small_drop

#First create scale variable for salary variable
def categorical_to_scale(df, var):
    new_df = df.copy()
    unique_val = np.unique(df[var])

    new_df['func_band'] = [2 if df[var][x] == unique_val[0] else 1 if df[var][x] == 
                          unique_val[1] else 0 for x in range(0,len(df[var]))]
    
    return new_df

#Undo scaling on target variable for final output
def undo_var_scaling(df, var, cat, new_col_name, unqiue_override=[], drop = False):
    new_df = df.copy()
    if len(unqiue_override) > 0:
        unique_val = unqiue_override
    else:
        unique_val = np.unique(df[var])  

    new_df[new_col_name] = [cat[0] if x == unique_val[0] else cat[1] if x == unique_val[1] 
                            else cat[2] for x in df[var]]  
    if drop == True:
        new_df = new_df.drop(var, axis = 1)

    return new_df

#For categoricals with too many unique vaues, instead will look at which are empty or nots
def set_empty(df, cat_var):   
    empty_var = [0 if x == 'none' else 1 for x in df[cat_var]]
    new_var_name = cat_var + "_empty"
    df[new_var_name] = empty_var
    return df

#Encodes categorical variables
def onehot_encode(df, override = []):
    if len(override) >= 1:
        new_df = df.copy()
        cat = override
        new_df = new_df[[x for x in new_df if x not in cat]]
    else:
        numericals = df.get(numerical_features(df))
        new_df = numericals.copy()
        cat = categorical_features(df)
    for categorical_column in cat:
        new_df = pd.concat([new_df, pd.get_dummies(df[categorical_column], prefix = categorical_column)], 
                axis=1)     
    return new_df

#Small multiples of numerical value histograms
def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure(figsize=(15,8))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()
    plt.show()

#Reduce the number of categories by grouping those below a certain percentile
def group_underrepresented_cat(df, var, tol = 0.001):
    for cat in df[var].unique():
        cat_percentile = len(df.loc[df[var] == cat,var])/len(df[var])
        if cat_percentile < tol:
            df.loc[df[var] == cat,var] = 'other'
    return df

def group_mean(df, group_vars, target_vars, verbose = False):
    #Compute population, long, lats, and gps_height averages in districts within regions
    region_district_mean = df.groupby([group_vars[0],group_vars[1]]).mean()
    region_district_mean_1 = pd.DataFrame(region_district_mean[target_vars[0]])
    region_district_mean_2= pd.DataFrame(region_district_mean[target_vars[1]])
    region_district_mean_3 = pd.DataFrame(region_district_mean[target_vars[2]])
    region_district_mean_4 = pd.DataFrame(region_district_mean[target_vars[3]])

    #Map based on district and region (multi-index dataframe)
    for row in range(len(df)):
        #Population
        if df.loc[row, target_vars[0]] == 0:
            df.loc[row, target_vars[0]] = region_district_mean_1.loc[(df.loc[row, group_vars[0]],
                                            df.loc[row, group_vars[1]]), target_vars[0]]
        #Latitude
        if df.loc[row, target_vars[1]] == 0:
            df.loc[row, target_vars[1]] = region_district_mean_2.loc[(df.loc[row, group_vars[0]],
                                            df.loc[row, group_vars[1]]), target_vars[1]]
        #Longitude
        if df.loc[row, target_vars[2]] == 0:
            df.loc[row, target_vars[2]] = region_district_mean_3.loc[(df.loc[row, group_vars[0]],
                                            df.loc[row, group_vars[1]]), target_vars[2]]

        #GPS height
        if df.loc[row, target_vars[3]] == 0:
            df.loc[row, target_vars[3]] = region_district_mean_4.loc[(df.loc[row, group_vars[0]],
                                            df.loc[row, group_vars[1]]), target_vars[3]]
    if verbose == True:
        print(df.head())
    return df

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

#Standardization for Polynomial Feature
def standardize2(df):
    standardized_numericals = preprocessing.scale(df)
    df = standardized_numericals  
    return df

#Clip outliers based on number of standard deviations
def sigmaclip (input, min, max):
    mean = np.mean(input)
    stddev = np.std(input)
    return np.clip(
        input,
        mean - min * stddev,
        mean + max * stddev
    )

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
    classifier = RandomForestClassifier(random_state = seed)
    classifier.fit(X_train, y_train)   
    return classifier.score(X_test, y_test)

def cv_evaluate(df, target_var, seed, cv):
    # Create Random Forest object
    lm = RandomForestClassifier(random_state = seed)
    kfolds = KFold(n_splits=cv, shuffle=True, random_state=seed)

    X = df.drop([target_var], axis=1)
    y = df.func_band.reset_index(drop=True)
    benchmark_model = make_pipeline(RobustScaler(), lm).fit(X=X, y=y)
    scores = cross_val_score(benchmark_model, X, y, scoring='accuracy', cv=kfolds)   
    return scores[scores >= 0.0]

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

#Genetic Program Features
def gp_features(df, target, random_state, generations = 5, function_set = ['add', 'sub', 'mul', 'div']):
    X = df.loc[:,(df.columns != target)]
    y = df.loc[:, target]   
    
    gp = SymbolicTransformer(generations=generations, population_size=1000,
                         hall_of_fame=100, n_components=12,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=0,
                         random_state=random_state, n_jobs=-1)
    gp.fit(pd.get_dummies(X), y)
    df = gp_transform(df, gp.transform, X)
    
    return df, gp.transform

#Transform data using input genetic transformer
def gp_transform(df, transformer,X):
    gp_features = transformer(pd.get_dummies(X))
    feats = pd.DataFrame(gp_features)
    feats.columns = ['gp{}'.format(i) for i in range(len(list(feats)))]
    df = pd.concat([df, feats], axis = 1)
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

#Calculates the operation years of a well
def operation_years(df):
    df['operation_year'] = df['Year'] - df['construction_year']
    df.loc[df['operation_year'] < 0, 'operation_year'] = 0
    return df

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def distance_from_capital(df, capital_lon, capital_lat):
    distance =[]
    for x in range(len(df)):
        distance.append(haversine(capital_lon, capital_lat, df.loc[x,'longitude'],df.loc[x,'latitude']))

    df['distance'] = distance
    return df

def test_feature_engineering(target, model, X, y, n_splits, random_state, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = model.fit(X_train,y_train.values.ravel())

    df = X.join(y)
    print('Accuracy of Feature: {:.3f}'.format(model.score(X_test, y_test.values.ravel())))
    accuracy = cv_evaluate(df, target, random_state, cv = n_splits)
    print('Mean Accuracy after CV: {:.3f} +/- {:.03f}'.format(np.mean(accuracy), np.std(accuracy)))
    print('Best Accuracy after CV: {:.3f}'.format(max(accuracy)))

    return df

#Tune model and give output based on given parameters
def tune_model(estimator, param, n_jobs, X_train, y_train, random_state, scoring_metric = 'accuracy', cv = 5, 
               gridSearch = False, verbose = False):
    kfolds = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    if gridSearch == True:
        gsearch = GridSearchCV(estimator = estimator, param_grid = param, 
                                    scoring=scoring_metric,n_jobs=n_jobs, cv=kfolds)
    else:
        gsearch = RandomizedSearchCV(estimator = estimator, param_distributions = param, 
                                    scoring=scoring_metric,n_jobs=n_jobs, cv=kfolds)
    gsearch.fit(X_train, y_train)
    tuned_model= gsearch.best_estimator_ 

    if(verbose == True):
        print('='*20)
        print("best params: " + str(gsearch.best_estimator_))
        print("best params: " + str(gsearch.best_params_))
        print('best score:', gsearch.best_score_)
        print('='*20)

    return tuned_model

#Prepare and save model results in required csv format
def csv_conversion(df, y, old_target_name, new_target_name,target_cat, unqiue_override=[]):
    #Merge prediction with original data set to map with id
    raw_output = df.join(y)
    raw_output = raw_output.loc[:,['id', old_target_name]]

    #Undo scaling of target variable to revert to original format
    clean_output = undo_var_scaling(raw_output, old_target_name, new_col_name = new_target_name, 
                                    unqiue_override = unqiue_override, cat = target_cat, drop = True)
    return clean_output

#Ensure final test set matches train data set
def column_sync(X_test, X_test_col, X_train_col):
    X_baseline_test = X_test.copy()
    for x in range(0,min(len(X_train_col),len(X_test_col))):
        if X_train_col[x] != X_test_col[x] and X_train_col[x] not in X_test_col:
            X_baseline_test[X_train_col[x]] = 0 * len(X_baseline_test.index)
        if X_train_col[x] != X_test_col[x] and X_test_col[x] not in X_train_col:
            X_baseline_test = X_baseline_test.drop(X_test_col[x], axis = 1)
    return X_baseline_test

####################################### Not Applicable for this Project #######################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

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

#Remove engineered features that resulted in invalid entries
def NaN_removal(df):
    drop_cols = [x for x in df if df[x].isnull().sum() == len(df)]
    output_df = df.drop(drop_cols, axis = 1)
    print('Need to remove {} columns with invalid entries.'.format(len(drop_cols)))
    return output_df

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

#Gets the distribution of given list of booleans
def boolean_dist(df, bools):
    output = []
    for col in bools:
        dist = df[col].value_counts()
        output.append(dist)
    return output

def feature_reduction(model, score_target, cv_input, X_entire_set, X_train_set, y_train_set):
    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(model, step=1, cv = cv_input, scoring=score_target)
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

#Plot Two Dimensional PCA graph
def pca_analysis(transformed_data, target, pca_1, pca_2, labels, labl):
    cdict={0:'red',1:'yellow',2:'green'}
    marker={0:'x',1:'*',2:'o'}
    alpha={0:.5, 1:.5, 2:.2}
    fig,ax=plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')
    for l in np.unique(labels):
        ax.scatter(transformed_data.loc[transformed_data[target] == l, pca_1],
            transformed_data.loc[transformed_data[target] == l, pca_2],c=cdict[l],s=40,label=labl[l],
            marker=marker[l],alpha=alpha[l])

    plt.xlabel("PCA_{}".format(pca_1),fontsize=14)
    plt.ylabel("PCA_{}".format(pca_2),fontsize=14)
    plt.legend()
    plt.show()