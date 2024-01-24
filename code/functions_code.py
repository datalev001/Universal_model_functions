import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from scipy.stats import pointbiserialr
import os
from scipy.stats import chi2_contingency
from scipy.stats import pointbiserialr, f_oneway, kruskal
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import datetime, timedelta
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

##################some tools to be used##########################################
def makerkgrp(D, gp):
    if D['grp'] >= gp:
        x = gp - 1
    else:
        x = D['grp']
        
    return x   

def remove_sub_list(lst, sub_lst, dup):
    if dup:
        filter_set = set(sub_lst)
        lst_fixed = [x for x in lst if x not in filter_set]
    else:    
        lst_fixed = list(set(lst) - set(sub_lst))
        
    return lst_fixed     

def cosemerge(dflist, keys, stymerge):
    k = 0    
    res_df = pd.DataFrame([])
    for tem_df in dflist:
        if k==0: res_df = tem_df.copy()
        else: res_df = pd.merge(res_df, tem_df, on = keys, how = stymerge)
        k = k + 1
    return res_df
    
#################################################################
'''
1 ) Function Purpose:
The get_corr_with_target function serves for feature selection by calculating the correlation between a target variable and a list of variables in a DataFrame.

Parameters:
target: The target variable for correlation analysis.
dataframe: The DataFrame containing the variables.
variable_names: List of variable names to be correlated with the target.
correlation_threshold: Absolute correlation threshold for variable selection.
Output:
The function returns a DataFrame with columns 'varname', 'correlation', 'abscorr', 'order'. It represents variables sorted by their absolute correlation with the target, filtered by the specified threshold.
'''

def getcorr(target, DF, variable_names, correlation_threshold):
    corr = list()
    for vname in variable_names: 
        X = DF[vname]              
        C = np.corrcoef(X, target)      
        beta = np.round(C[1, 0], 4)
        corr = corr + [beta] 
    
    corrdf = pd.DataFrame({'varname': variable_names, 'correlation': corr})
    corrdf['abscorr'] = np.abs(corrdf['correlation'])
    corrdf.sort_values(['abscorr'], ascending=False, inplace=True)
    seq = range(1, len(corrdf) + 1)
    corrdf['order'] = seq
    corrdf['abscorr'] = corrdf['abscorr'].fillna(0.0)
    corrdf = corrdf[corrdf.abscorr >= correlation_threshold]
    return corrdf

'''
example: 
    
df = pd.read_csv('tant_train.csv')
df.dtypes
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
df_num = df_num.fillna(df_num.mean())
varnamelist = list(df_num.columns)
varnamelist.remove('Survived')
cor_df = getcorr(df.Survived, df, varnamelist, 0.02)

  varname  correlation  abscorr  order
1  Pclass      -0.3385   0.3385      1
5    Fare       0.2573   0.2573      2
4   Parch       0.0816   0.0816      3
3   SibSp      -0.0353   0.0353      4

'''

###########################################################

'''
2) Function Purpose:
The getcorr_binary_target function calculates and returns the Pearson correlation, Kendall correlation, and Kolmogorov-Smirnov (KS) statistic for a binary target variable (Y) with a list of independent variables (varnamelist) in a DataFrame (df). It filters the variables based on the absolute Pearson correlation exceeding a specified threshold (thresh).

Parameters:
Y: Binary target variable.
df: DataFrame containing the variables.
varnamelist: List of variable names to be correlated with the binary target.
thresh: Threshold for absolute Pearson correlation to filter variables.
Output:
The function returns a DataFrame (corrdf) containing:

varname: Variable names from varnamelist.
pearson_correlation: Pearson correlation coefficients.
kendalltau_correlation: Kendall correlation coefficients.
ks_statistic: Kolmogorov-Smirnov statistics.
abscorr: Absolute values of Pearson correlation coefficients.
order: Ranking order based on absolute Pearson correlation values.
'''

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, ks_2samp

def getcorr_plus(target, DF, variable_names):
    corr = []
    kendall_corr = []
    ks_statistic = []
    for vname in variable_names:
        X = DF[vname]
        # Calculate Pearson correlation
        pearson_corr = np.corrcoef(X, target)[1, 0]
        corr.append(np.round(pearson_corr, 4))
        # Calculate Kendall correlation
        kendall_corr.append(np.round(kendalltau(X, target).correlation, 4))
        # Calculate KS statistic
        ks_statistic.append(ks_2samp(X[target == 1], X[target != 1]).statistic)

    corrdf = pd.DataFrame({'varname': varnamelist, 'pearson_correlation': corr,
                           'kendalltau_correlation': kendall_corr, 'ks_statistic': ks_statistic})
    
    corrdf['abscorr'] = np.abs(corrdf['pearson_correlation'])
    corrdf.sort_values(by='abscorr', ascending=False, inplace=True)
    corrdf['order'] = range(1, len(corrdf) + 1)
    corrdf['abscorr'] = corrdf['abscorr'].fillna(0.0)
    
    return corrdf

'''
example: 
    
DF = df.copy()
varnamelist = ['Pclass','Fare', 'Parch', 'SibSp', 'Age']
df[varnamelist] = df[varnamelist].fillna(df[varnamelist].median())
cor_df = getcorr_plus(DF.Survived, DF, varnamelist) 

'''


###################################################

'''
3) Function Purpose:
The createdummy function converts categorical features into dummy variables, considering only the top high-frequency levels specified by the tops parameter.

Parameters:
df (DataFrame): Input DataFrame.
catvars (list): List of categorical feature column names.
tops (int): Number of top high-frequency levels to choose for each categorical variable.
Set tops = 0 to convert all levels.
Output:
DataFrame containing dummy variables converted from the top high-frequency levels of categorical features.
'''

def create_dummy(df, catvars, tops):

    def assignval(data, field, lst):
        if (data[field] in lst) == False:
            x = 'RESTCAT'
        else:
            x = data[field]
        return x     
    
    res_df = pd.DataFrame([])
    data_df = df.copy()

    for itv in catvars:
        data_df[itv] = data_df[itv].fillna('MISS')

        df_cnt = data_df[itv].value_counts().reset_index()
        df_cnt.columns = [itv, 'cnt']
        df_cnt = df_cnt.sort_values(['cnt'], ascending=False)

        if tops > 0:
            df_cnt_hd = df_cnt.head(tops)
            itvlist = list(df_cnt_hd[itv])
            data_df[itv] = data_df.apply(assignval, args = (itv, itvlist), axis = 1)
                    
        mv_df = pd.get_dummies(data_df[itv])
        mv_df.columns = ['indic_' + itv + '_' + str(it) for it in mv_df.columns]
        res_df = pd.concat([res_df, mv_df], axis=1)

    return res_df

'''
Example:
    
df.shape 
df.dtypes   
catvars = ['Cabin', 'Embarked']
len(set(df['Cabin']))
df.Cabin.value_counts()
df.Embarked.value_counts()
data_dummy = create_dummy(df, catvars, 5)
list(data_dummy.columns)
data_dummy.shape

'''

##############################################################

'''
4) Function Purpose:
The getdummy_corr function creates dummy variables from categorical features and selects those with higher absolute correlation with the target variable.

Parameters:
df (DataFrame): Input DataFrame.
target (str): Column name of the target variable.
corr_threshold (float): Threshold for absolute correlation.
catvars (list): List of categorical feature column names.
tops (int): Number of top high-frequency levels to choose for each categorical variable.
Set tops = 0 to convert all levels.
Output:
DataFrame containing dummy variables selected based on their absolute correlation with the target variable.
'''

import numpy as np
import pandas as pd

def getdummy_corr(df, target, corr_threshold, catvars, tops):

    dummy_df = create_dummy(df, catvars, tops)

    corr_df = getcorr(df[target], dummy_df, dummy_df.columns, 0)

    selected_dummies = corr_df[corr_df['abscorr'] >= corr_threshold]

    return dummy_df[selected_dummies['varname']]

'''
Example:
    
target, corr_threshold, catvars, tops = 'Survived', 0.03, ['Cabin', 'Embarked', 'Sex'], 5  
    
data_dummy_sel  = getdummy_corr(df, target, corr_threshold, catvars, tops)
list(data_dummy_sel.columns)
'''


###########################################################

'''
5) Function Purpose:
The cal_CramerV function calculates Cramer's V, a measure of association, between each categorical feature (cat_features) and a target variable (target) in a modeling dataset. It returns the features and their corresponding Cramer's V values, aiding in feature selection.

Parameters:
data_df: DataFrame containing the dataset.
target: Target variable for Cramer's V calculation.
cat_features: List of categorical features to be evaluated.
Output:
The function returns a DataFrame (CRV_DF) with columns:

feature: Categorical feature names.
value_gain: Cramer's V values representing the association with the target variable.

'''


def cal_CramerV(data_df, target, cat_features):
    rec_lst = []

    for feature in cat_features:
        data_crosstab = pd.crosstab(data_df[feature], data_df[target], margins=False)
        X2, _, _, _ = chi2_contingency(data_crosstab, correction=False)
        N = np.sum(data_crosstab.values)
        minimum_dimension = min(data_crosstab.shape) - 1
        # Calculate Cramer's V, range = [0, 1]
        CRV = np.sqrt((X2 / N) / minimum_dimension)
        rec_lst.append([feature, CRV])

    CRV_DF = pd.DataFrame(rec_lst, columns=['feature', 'value_gain'])
    return CRV_DF

'''
Example:
product_preference_df = pd.read_csv('product_preference.csv')
CramerV_df = cal_CramerV(product_preference_df, 'target', ['product_type', 'brand', 'price_layer'])
'''

#################################################################

'''
6) Function Purpose:
The impurity_values function calculates the impurity (Gini) of each feature in a modeling dataset and returns the features with impurity values below a specified threshold. This aids in feature selection for modeling.

Parameters:
data_df: DataFrame containing the dataset.
features: List of features to calculate impurity.
target: Target variable for impurity calculation.
target_type: Type of the target variable ('B' for binary, 'C' for continuous, 'M(CAT)' for multi-class categorical).
threshold: Threshold for impurity value to filter features.
Output:
The function returns a DataFrame (impurity_df) with columns:

type: Type indicator ('IMP_VAL').
feature: Feature names.
value_gain: Impurity values representing the Gini index.

'''

def impurity_values(data_df, features, target,
                    target_type, threshold, threshold_len):
    
    def gini_impurity(data, target):
        value_counts = data[target].value_counts(normalize=True)
        return 1 - (value_counts ** 2).sum()
    
    def arrange_bin(data_df, feature):   
        df = data_df.copy()
        
        set_n = len(set(df[feature]))
        q_v = 2
        if set_n < 10: 
            df[feature] = df[feature] + 0
        elif set_n < 40:     
            q_v = 3
        elif set_n < 200:      
            q_v = 6
        elif set_n < 2000:      
            q_v = 15
        else:    
            q_v = 30
        
        if set_n > 10:
            member = pd.qcut(df[feature], q = q_v, duplicates='drop')
            qcut = pd.qcut(df[feature], q = q_v, duplicates='drop').reset_index()
            k = list(set(qcut[feature].astype(str)))
            v = [itv for itv in range(1, len(k) + 1)]
            m = dict(zip(k, v))
            df[feature] = member.astype(str).map(m)
            
        return df

    lbl = preprocessing.LabelEncoder()

    df = data_df.copy()
    target_miss = df[target].isnull().sum()
    
    if target_miss > 0 or len(df) <= 20:
        return pd.DataFrame(columns=['type', 'feature', 'value_gain'])

    if target_type in ['B', 'M(CAT)']:
        df['target_new'] = df[target]
    elif target_type in ['C', 'M(ORD)']:
        quantiles = df[target].quantile([0.01 * j for j in range(1, 99)])
        criteria = [
        (
            (df[target] > quantiles.iloc[j]) 
            if pd.notna(quantiles.iloc[j]) 
            else True
        ) & (
            (df[target] <= quantiles.iloc[j + 1])
            if pd.notna(quantiles.iloc[j + 1])
            else True
        ) 
        for j in range(len(quantiles) - 1)
        ]
                
        th = threshold_len * len(df)
        df['target_new'] = np.select(criteria, range(1, len(quantiles)), 0)
        tmp = df['target_new'].value_counts().reset_index()
        rmclist = list(tmp[tmp.target_new > th]['index'])
        df = df[df.target_new.isin(rmclist)]

    gain_lst = []

    for f in features:
        df[f] = df[f].fillna(df[f].mean()) if df[f].dtype in numerics else df[f].fillna(str(f) + '_MISSING')
        df[f] = lbl.fit_transform(df[f].astype(str))
        df = arrange_bin(df, f)

        cat_feature_levels = set(df[f])
        cat_feature_len = len(cat_feature_levels)

        if cat_feature_len > 0:
            gini = sum(len(df[df[f] == feature_item]) / len(df) * gini_impurity(df[df[f] == feature_item], 'target_new') for feature_item in cat_feature_levels) / cat_feature_len
            vc = [f, gini]
            gain_lst.append(vc)
        else:
            gini = 0

    impurity_df = pd.DataFrame(gain_lst, columns=['feature', 'value_gain'])
    impurity_df['type'] = 'IMP_VAL'
    impurity_df = impurity_df[['type', 'feature', 'value_gain']].sort_values(by='value_gain')
    impurity_df = impurity_df[impurity_df.value_gain < threshold]
    return impurity_df

'''
Example:
product_preferenceII = pd.read_csv('product_preferenceII.csv')

features = ['product_type' , 'brand', 'price_layer']
target, target_type, threshold, threshold_len = 'target_B', 'B', 1, 0.003
purity_df = impurity_values(product_preferenceII, features, target, 
                              target_type, threshold, threshold_len)

target, target_type, threshold, threshold_len = 'target_c','C', 10, 0.003
purity_df = impurity_values(product_preferenceII, features, target, 
                              target_type, 1, threshold_len)

target, target_type, threshold, threshold_len =  'target' ,'M(CAT)', 10, 0.003
purity_df = impurity_values(product_preferenceII, features, target, 
                              target_type, 1, threshold_len)

target, target_type, threshold, threshold_len =  'target_o' ,'M(ORD)', 10, 0.003
purity_df = impurity_values(product_preferenceII, features, target, 
                              target_type, 1, threshold_len)

'''


'''
7) Function Purpose:
The multiCorr function identifies and removes features with high correlation coefficients from a dataset, facilitating the selection of relatively independent independent variables for modeling. The threshold parameter mulcol_limit specifies the maximum absolute correlation allowed.

Parameters:
data_df: DataFrame containing the dataset.
features: List of feature names to check for multicollinearity.
mulcol_limit: Threshold for maximum absolute correlation coefficient allowed.
cor_method: Method to compute correlation coefficients.
Output:
The function returns a list containing:

corrMat: Correlation matrix of the remaining features.
features_remain: List of final independent variables with low pairwise correlation.
removed_features: List of features removed due to high correlation with others.
'''

def multiCor(data_df, features, mulcol_limit, cor_method):
    features_remain = features.copy()
    df_dm_cp = data_df[features_remain].copy()
    
    removed_features = set()

    while True:
        corrMat = df_dm_cp.corr(method=cor_method).abs()
        np.fill_diagonal(corrMat.values, 0)  # Exclude diagonal elements
        high_corr_pairs = np.column_stack(np.where(corrMat > mulcol_limit))
        
        if len(high_corr_pairs) == 0:
            break
        
        correlated_features = set(features_remain[i] for pair in high_corr_pairs for i in pair)
        features_remain = [feat for feat in features_remain if feat not in correlated_features]
        removed_features.update(correlated_features)
        df_dm_cp = df_dm_cp[features_remain]
    
    corrMat = corrMat.fillna(0)    
    return [corrMat, features_remain, list(removed_features)]

def multiCorr(data_df, features, mulcol_limit, cor_method):
    def remove_sub_list(lst, sub_lst):
        lst_fixed = list(set(lst) - set(sub_lst))
        return lst_fixed
        
    features_remain = features[:]    
    
    df_dm_cp = data_df.copy() 
    corrMat = df_dm_cp[features_remain].corr(method = cor_method)
    LSS = list(corrMat.columns)
    revnames_all = []
    revnames = []
    for colv in LSS:
        LS = list(corrMat.columns)
        L = len(LS)
        for j in range(L):
            col = LS[j]
            corv = (abs(corrMat[col])>mulcol_limit).sum()
            if corv > 1:
                z = corrMat.loc[abs(corrMat[col])>mulcol_limit][col]
                if len(z)>0:
                    revnames = list(dict(z).keys())
                    if col in revnames:
                        revnames.remove(col)
                    if len(revnames) > 0:
                        revnames_all.extend(revnames)   
                        df_dm_cp = df_dm_cp.drop(revnames, axis = 1)
                        
                        features_remain = remove_sub_list(features_remain, revnames)
                        corrMat = df_dm_cp[features_remain].corr(method = cor_method)
                        break
                    
    revnames_all = list(set(revnames_all))
    corrMat = corrMat.fillna(0)
    return [corrMat, features_remain, revnames_all]  

'''
example: 
    
df = pd.read_csv('tant_train.csv')
df.dtypes
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics).fillna(0)

df_obj = df.select_dtypes(include=obs)
data_dummy = createdummy(df, catvars, 5)
df_combine = pd.concat([df_num, data_dummy], axis = 1)
df_combine.isnull().sum()

cols = list(df_combine.columns)
cols.remove('PassengerId')
cols.remove('Survived')

# Example Usage
data_df, features, mulcol_limit, cor_method = df_combine.copy(), cols[:], 0.15, 'spearman' 
features_remain1 = multiCor(data_df, features, mulcol_limit, cor_method)[1]
len(features_remain1)
corrMat = multiCor(data_df, features, mulcol_limit, cor_method)[0]
corrMat.isnull().sum()


df = pd.read_csv('tant_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics).fillna(0)
df_obj = df.select_dtypes(include = ['object'])
catvars = ['Sex', 'Embarked', 'Pclass']
data_dummy = create_dummy(df, catvars, 5)
df_combine = pd.concat([df_num, data_dummy], axis = 1)
cols = list(df_combine.columns)
cols.remove('PassengerId')
cols.remove('Survived')

data_df, features, mulcol_limit, cor_method = df_combine.copy(), cols[:], 0.15, 'spearman' 
features_remain = multiCorr(data_df, features, mulcol_limit, cor_method)[1]
corrMat = multiCorr(data_df, features, mulcol_limit, cor_method)[0]
print (features_remain)

'''

#################################################

'''
8 ) Function Purpose:
The calc_iv_new function calculates the Information Value (IV) for a categorical feature with respect to a target variable.

Parameters:
df (DataFrame): Input DataFrame.
feature (str): Column name of the categorical feature.
target (str): Column name of the target variable.
pr (float): Minimum count ratio threshold for category selection.
catnum_th (int): Maximum number of top categories to consider.
Output:
A list containing IV value, qualified items, unqualified items, and a DataFrame with IV details for each category.
'''

def calc_iv(df, feature, target, pr=0.0003, catnum_th=50):
    df['cnt'] = 1
    df_sum = df.groupby([feature])[target, 'cnt'].sum().reset_index()
    df_sum['non_target'] = df_sum['cnt'] - df_sum[target]
    df_sum['cnt_ratio'] = df_sum['cnt'] / df_sum['cnt'].sum()

    df_sum_qualify = df_sum[(df_sum['cnt_ratio'] > pr) & (df_sum[target] > 0) & (df_sum['non_target'] > 0)]

    if len(df_sum_qualify) > 0:
        df_sum_qualify = df_sum_qualify.sort_values(['cnt_ratio'], ascending=False).head(catnum_th)
        
        tot_target = df_sum_qualify[target].sum()
        tot_no_target = df_sum_qualify['non_target'].sum()
        df_sum_qualify['non_target_rate'] = df_sum_qualify['non_target'] / tot_no_target
        df_sum_qualify['target_rate'] = df_sum_qualify[target] / tot_target
        df_sum_qualify['woe'] = np.log(df_sum_qualify['non_target_rate'] / df_sum_qualify['target_rate'])
        df_sum_qualify['iv'] = (df_sum_qualify['non_target_rate'] - df_sum_qualify['target_rate']) * df_sum_qualify['woe']
        IV = df_sum_qualify['iv'].sum()
        qualify_items = list(df_sum_qualify[feature])
        unqualify_items = list(set(df[feature]) - set(qualify_items))
    else:
        IV = -1
        qualify_items = []
        unqualify_items = list(set(df[feature]))
        df_sum_qualify = pd.DataFrame([])

    return [IV, qualify_items, unqualify_items, df_sum_qualify]

'''
example

df = pd.read_csv('tant_train.csv')
df.dtypes
df.shape
df[['Pclass', 'Age', 'Cabin', 'Embarked']].value_counts()
df[['Pclass', 'Age', 'Cabin', 'Embarked']].isnull().sum()

feature, target = 'Cabin', 'Survived'
df[feature] = df[feature].fillna('UNKN')
Cabin_iv = calc_iv(df, feature, target, pr=0.000, catnum_th=50)[0]
res_df = calc_iv(df, feature, target, pr=0.000, catnum_th=50)[3]

'''

################################################
'''
9) Function Purpose:
The corr_cat_target function calculates correlations between a categorical variable and a continuous target using three methods: Point-Biserial Correlation Coefficient, Eta-coefficient from one-way ANOVA, and Kruskal-Wallis H-statistic. It returns the correlation results in a list of dictionaries.

Parameters:
DF (DataFrame): The input DataFrame.
feature (str): The name of the categorical variable column.
target (str): The name of the continuous target column.
Output:
A list containing dictionaries with correlation results from the three methods for each category in the categorical variable.

'''


def corr_cat_target(DF, feature, target):
    # 1) Point-Biserial Correlation Coefficient for each category
    res_r = []
    cats = set(DF[feature])
    r, p_value = 0, 0
    DF_dummy = pd.get_dummies(DF[feature], drop_first=True) + 0
    cols = list(DF_dummy.columns)
    n = 0
    
    for it in cols:
        r0, p_value0 = pointbiserialr(DF_dummy[it], DF[target])
        print (r0)
        r, p_value = r + abs(r0), p_value + abs(p_value0)
        n = n + 1

    r_res, p_value_res = r/n, p_value/n
    dicv = {'Point-Biserial corr': r_res, 'p_value': p_value_res}
       
    # 2) Eta-coefficient using one-way ANOVA
    f_statistic, p_value_anova = f_oneway(*[DF[DF[feature] == cat][target] for cat in cats])
    eta_squared = f_statistic / (f_statistic + (len(DF[target]) - 1))
    dic_anova = {'eta_coef': round(eta_squared**0.5, 3), 'p_value_anova': p_value_anova}

    # 3) Kruskal-Wallis H-statistic
    h_statistic, p_value_kruskal = kruskal(*[DF[DF[feature] == cat][target] for cat in cats])
    dic_kruskal = {'h_statistic': h_statistic, 'p_value_kruskal': p_value_kruskal}

    return [dicv, dic_anova, dic_kruskal]

'''
Example:
product_price_df = pd.read_csv('product_price.csv')
# Test the corr_cat_target function
DF, feature, target = product_price_df.copy(), 'product_type', 'price'
result = corr_cat_target(DF, feature, target)
print(result)
'''

#############################################

'''
10) Function Description:

The find_vif function calculates the Variance Inflation Factors (VIFs) for a given set of numeric variables in a DataFrame. VIF is a measure of the extent to which the variance of a regression coefficient is increased due to collinearity in the data. High VIF values indicate a high degree of multicollinearity.

Parameters:

df (pd.DataFrame): The input DataFrame containing the relevant numeric columns.
features (list): A list of numeric column names for which VIFs need to be calculated.
Returns:

pd.DataFrame: A DataFrame with two columns, 'varname' and 'vif', where 'varname' contains the column names and 'vif' contains their corresponding VIF values. The DataFrame is sorted in descending order by VIF.

Interpret Results:

vif_result will contain a DataFrame with columns 'varname' and 'vif'.
Sort the DataFrame by 'vif' in descending order to identify variables with higher VIFs.
Higher VIF values (typically above 5 or 10) indicate potential multicollinearity issues.

Variables with higher VIF values may need closer examination, as they could be correlated with other predictors.
Consider removing or combining variables with high VIF to reduce multicollinearity.
Lower VIF values (closer to 1) suggest less collinearity and are generally preferred for robust regression modeling.
'''

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def find_vif(data_df, features):

    df_numeric = data_df[features]

    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["varname"] = features
    vif_data["vif"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

    # Sort by VIF in descending order
    vif_data = vif_data.sort_values(by="vif", ascending=False).reset_index(drop=True)

    return vif_data

'''
Example:
    
df = pd.read_csv('tant_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
df_num = df_num.fillna(df_num.median())
cols = list(df_num.columns)
cols.remove('PassengerId')
cols.remove('Survived')
data_df, features = df_num.copy(), cols[:]
df_combine.isnull().sum()
vif_df = find_vif(data_df, features)

'''
######################################

'''
11 )Function Purpose:
The woefeatures function calculates the Weight of Evidence (WOE) for a categorical feature in a dataframe.

Parameters:
df (DataFrame): Input DataFrame.
ID (str): Column name of the unique identifier.
target (str): Column name of the target variable.
feature (str): Column name of the categorical variable.
splitseed (int): Random seed for splitting data into train and test.
woename (str): Desired name for the WOE column in the resulting dataframe.
Output:
A DataFrame containing the original data along with the added WOE values for the specified feature.
'''

def woefeatures(df, ID, target, feature, splitseed, woename):

    df_out = df[[ID, target, feature]]

    X_train, X_test, _, _ = train_test_split(df, df[target], test_size=0.5, random_state=splitseed)

    def calculate_woe_iv(data):
        woe_res = (pd.crosstab(data[feature], data[target], normalize='columns')
                .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                .assign(iv=lambda dfx: np.sum(dfx['woe'] * (dfx[1] - dfx[0])))).reset_index()
        
        woe_res.rename(columns = {'woe' : woename}, inplace = True)
        return woe_res

    df_woe_iv_train = calculate_woe_iv(X_train)
    df_woe_iv_test = calculate_woe_iv(X_test)
    cols = list(df_woe_iv_train.columns)
    cols.remove(feature)

    df_woe_iv = pd.concat([df_woe_iv_train, df_woe_iv_test])
    df_woe_iv = df_woe_iv.fillna(0)
    df_woe_iv = df_woe_iv.groupby(feature)[cols].mean().reset_index()
    df_woe = df_woe_iv[[feature, woename]]
    df_out = pd.merge(df_out, df_woe, on=[feature], how='left')
    df_out = df_out.fillna(df_out.median())

    return df_out

'''
Example:
df = pd.read_csv('tant_train.csv')
df.dtypes
df.Parch.value_counts()
df.Embarked.value_counts()
df.Embarked.isnull().sum()
df.SibSp.value_counts()
df.Age.value_counts()

df['Age'] = df['Age'].fillna(df['Age'].median())
grp = sorted(list(set(pd.qcut(df.Age, q=5, duplicates = 'drop').astype(str))))
grp_dict = dict(zip(grp, ['agegrp_' + str(j) for j in range(1, len(grp) + 1)]))
df['Age_grp'] = pd.qcut(df.Age, q=5, duplicates = 'drop').astype(str).map(grp_dict)
df['Age_grp'].value_counts()

ID, target, feature, splitseed, woename = 'PassengerId', 'Survived', 'Age_grp', 12, 'age_woe'
woe_df = woefeatures(df, ID, target, feature, splitseed, woename)

'''

#####################################################################

'''
12) Function Purpose:
The createtmvar function extracts time-related variables from a date string column in a dataframe.

Parameters:
df (DataFrame): Input DataFrame.
ID (str): Column name of the unique identifier.
datev_str (str): Column name of the date string variable.
today_str (str): String representation of the ending date.
Output:
A DataFrame containing the original data along with additional time-related variables.

'''

def create_time_var(df, ID, datev_str, today_str):

    df['todayv'] = pd.to_datetime(today_str)
    df['datev'] = pd.to_datetime(df[datev_str], errors='coerce')

    df['weekday'] = df['datev'].dt.weekday + 1
    df['year'] = df['datev'].dt.year
    df['month'] = df['datev'].dt.month
    df['day'] = df['datev'].dt.day
    df['monthweek'] = (df['day'] / 9).round(0) + 1

    df['days_dist'] = (df['todayv'] - df['datev']).dt.days
    df['weekid'] = df['datev'].dt.week

    df['maxdate'] = df['datev'].max()
    df['mindate'] = df['datev'].min()
    df['maxdate_str'] = df['maxdate'].dt.strftime('%Y-%m-%d')
    df['mindate_str'] = df['mindate'].dt.strftime('%Y-%m-%d')

    featurelist = [ID, 'todayv', 'datev', 'weekday', 'year', 'month', 'day', 
                    'monthweek', 'days_dist', 'weekid', 'maxdate', 'mindate', 
                    'maxdate_str', 'mindate_str']

    df = df[featurelist]
    return df

'''
Example:
    
tran_df = pd.read_excel('retail.xlsx')
tran_df.dtypes
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity']>0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]
tran_df.InvoiceDate.median()

tran_df['transaction_date'] = tran_df['InvoiceDate'].dt.date.astype(str)
tran_df_sum = tran_df.groupby('transaction_date')['Quantity', 'Price'].sum().reset_index()
tran_df_sum['date_id'] = range(1, len(tran_df_sum) + 1)

df, ID, datev_str, today_str = tran_df_sum.copy(),'date_id', 'transaction_date', '2010-09-11' 
tran_df_sum_tm = create_time_var(df, ID, datev_str, today_str)

tran_df_sum_all = pd.merge(tran_df_sum, tran_df_sum_tm, on = ID, how= 'inner')
list(tran_df_sum_all.columns)
tran_df_sum_all['days_dist_grp'] = pd.qcut(tran_df_sum_all.days_dist, q=5, duplicates = 'drop').astype(str)

checklist =  ['weekday', 'month', 'monthweek', 'days_dist_grp']
for itv in checklist:
    print ('--------' + str(itv) + '----------')
    pattern_df = tran_df_sum_all.groupby(itv)['Quantity', 'Price'].mean().reset_index()
    print (pattern_df)
'''

#######################################################################

'''

13) Function Description:

The create_cat_feature function generates new features based on categorical and
 numeric variables in a DataFrame. It calculates counts and averages for each
 category within the categorical variables and matches them to the corresponding IDs.
 The resulting DataFrame includes ID and all the newly created features.

Parameters:

df (pd.DataFrame): The input DataFrame.
id (str): The name of the unique ID column in the DataFrame.
catvars (list): A list of names of categorical variables in the DataFrame.
numvars (list): A list of names of numeric variables in the DataFrame.
Returns:

pd.DataFrame: A DataFrame with ID and the newly created features based on 
counts and averages.

Interpret Results:

result_df will contain a DataFrame with 'id' and newly created features based on counts and averages.
Each categorical variable's count and numeric variable's average for each category will be used as features.


The `create_cat_feature` function generates new features based on the provided information. It performs the following tasks:

1. For each categorical variable in `catvars`, the function calculates the count for each category within the DataFrame (`df`). It matches each unique ID's category for that variable and records the matched count as a new feature. For instance, if, for a specific row with `id=1`, the `product_type` is 'A', and there are 100 records in the DataFrame with `product_type='A'`, the newly created feature for `id=1` would be `product_type_cnt = 100`. This process is repeated for all categorical variables in `catvars`.

2. For each categorical variable in `catvars` and each numeric variable in `numvars`, the function computes the average of the numeric variable for each category within the DataFrame. Similar to the previous step, it matches each unique ID's category for these variables and records the matched average as a new feature. For example, if, for a specific row with `id=1`, the `product_type` is 'A', and the average price is 32 in the DataFrame with `product_type='A'`, the newly created feature for `id=1` would be `product_type_avgprice = 32`. This process is repeated for all combinations of categorical variables in `catvars` and numeric variables in `numvars`.

Finally, the function returns a DataFrame containing the unique IDs (`id`) and all the created features above.

Usage:
```python
result_df = create_cat_feature(df, 'id', ['product_type', 'region'], ['price', 'quantity'])
print(result_df)

'''

def create_cat_feature(df, id_col, catvars, numvars, fillna=True):
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame({id_col: df[id_col].unique()})
    
    # 1. Calculate counts for each category within categorical variables
    for cat_var in catvars:
        vc_dict = df[cat_var].value_counts(normalize=True).to_dict()
        col_name = f'{cat_var}_cnt_ratio'
        result_df[col_name] = df[cat_var].map(vc_dict).astype('float32')

    # 2. Calculate averages for each category within numeric variables
    for num_var in numvars:
        for cat_var in catvars:
            new_col_name = f'{cat_var}_{num_var}_avg'
            temp_df = df.groupby([cat_var])[num_var].agg(['mean']).reset_index().rename(columns={'mean': new_col_name})
            temp_df.index = list(temp_df[cat_var])
            temp_dict = temp_df[new_col_name].to_dict()
            result_df[new_col_name] = df[cat_var].map(temp_dict).astype('float32')
            if fillna:
                result_df[new_col_name].fillna(-1, inplace=True)

    return result_df

'''
Example:
df = pd.read_csv('tant_train.csv')
df.dtypes
df.Fare
df.Parch.value_counts()
df.Pclass.value_counts()
df.Embarked.value_counts()
df.Embarked.isnull().sum()
df.SibSp.value_counts()
df.Age.value_counts()
df.isnull().sum()

df['Age'] = df['Age'].fillna(df['Age'].median())
df = df[df.Embarked.isnull() == False]


df = pd.read_csv('tant_train.csv')
numvars = ['Fare', 'Age']
catvars = ['Sex', 'Embarked', 'Pclass']
id_col = 'PassengerId'
res_df = create_cat_feature(df, id_col, catvars, numvars, fillna=True)
list(res_df.columns)
c = ['PassengerId', 'Sex_cnt_ratio', 'Embarked_cnt_ratio', 'Sex_Age_avg', 'Embarked_Fare_avg']
res_df[c].head(30)
'''

#####################################################################

'''
   14) Function Description
    Generates new features based on categorical and numeric variables in a DataFrame.
    Calculates counts and averages for each category within the categorical variables
    and matches them to the corresponding IDs. The resulting DataFrame includes ID and
    all the newly created features.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - id (str): The name of the unique ID column in the DataFrame.
    - catvars (list): A list of names of categorical variables in the DataFrame.
    - numvars (list): A list of names of numeric variables in the DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with ID and the newly created features based on counts and averages.

    Interpret Results:
    - result_df will contain a DataFrame with 'id' and newly created features based on counts and averages.
    - Each categorical variable's count and numeric variable's average for each category will be used as features.
'''

def cat_grp_agg_uniq(trans_data, cats, ID):
                                
    data_df = trans_data.copy()
    features = []
    uids_df = pd.DataFrame({ID: list(set(data_df[ID]))})
    
    for cat in cats:  
        unq_dict = data_df.groupby(ID)[cat].agg(['nunique'])['nunique'].to_dict()
        feature = 'uniq_'+ cat +'_cnt'
        uids_df[feature] = uids_df[ID].map(unq_dict).astype('float32')
        features.append(feature)
        
    res_df = uids_df[[ID] + features]    
    
    return res_df 

'''
Example:
    
tran_df = pd.read_excel('retail.xlsx')
tran_df.dtypes
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity']>0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]

tran_df = pd.read_excel('retail.xlsx')
trans_data, cats, ID = tran_df.copy(), ['Description'], 'Customer ID'
res_df = cat_grp_agg_uniq(trans_data, cats, ID)
'''

################################################################

'''
15) Define training and target data for building a repurchase prediction model.

Function Description:
The train_target_df function is designed to prepare training and target
 data for building a repurchase prediction model based on transaction data.
 It takes various parameters, including the input DataFrame, unique 
 identifier column name, transaction date column name, quantity column
 name, threshold for defining the target variable, performance date, 
 and performance days. The function returns two DataFrames: one for
 training data with various features, and the other for the target
 variable indicating whether a customer will repurchase or not.
 The function also allows customization of time ranges for
 window aggregated features.

The function uses aggregation and merging techniques to create features 
such as recency, early days, and window aggregated features based on 
the specified time ranges. The resulting DataFrames are structured and 
ready for use in predictive modeling.

  Parameters:
  - df (DataFrame): Input DataFrame containing transaction data.
  - id_col (str): Column name of the unique identifier.
  - trans_date (str): Column name of the transaction date.
  - qty (str): Column name of the quantity variable.
  - to_buy (int): Threshold quantity for defining the target variable.
  - performance_date (str): Date string representing the border date for predictor and target data.
  - performance_days (int): Number of days for the target performance window.
  - days_lst (list): List of time ranges for window aggregated features (default is [10, 30, 50]).

  Returns:
  - list: DataFrames containing training and target data.
'''


def train_target_df(df, id_col, trans_date, qty_list, to_buy, 
                    performance_date, performance_days, days_lst=[10, 30, 50]):
  
    def cosemerge(dflist, keys, stymerge):
        res_df = pd.DataFrame([])
        for i, tem_df in enumerate(dflist):
            if i == 0:
                res_df = tem_df.copy()
            else:
                res_df = pd.merge(res_df, tem_df, on=keys, how=stymerge)
        return res_df

    # Define training and target data
    df_tran_training = df[df[trans_date] <= performance_date]
    
    date_end = datetime.strptime(performance_date, "%Y-%m-%d") + timedelta(days=performance_days)
    date_end_str = datetime.strftime(date_end, "%Y-%m-%d")
    
    df_tran_target = df[(df[trans_date] > performance_date) & (df[trans_date] < date_end_str)]
    
    # Aggregating data for target
    target_df = df_tran_target.groupby(id_col)[qty_list].sum().reset_index()
    target_df['target'] = (target_df[qty_list].sum(axis=1) > to_buy).astype(int)
    target_df = target_df[[id_col, 'target']]
    
    # Create individual DataFrames for each feature
    feature_dfs = []
    
    for qty_col in qty_list:
        df_recency = df_tran_training.groupby(id_col).agg(
            Recency=(trans_date, lambda x: (pd.to_datetime(performance_date) - max(x)).days)
        ).reset_index()

        df_earlydays = df_tran_training.groupby(id_col).agg(
            EarlyDays=(trans_date, lambda x: (pd.to_datetime(performance_date) - min(x)).days)
        ).reset_index()

        df_frequency = df_tran_training.groupby(id_col).agg(Frequency=(qty_col, 'count')).reset_index()

        # Merge individual DataFrames based on 'Customer ID'
        dflist = [df_recency, df_earlydays, df_frequency]
        feature_df = cosemerge(dflist, [id_col], 'inner')
        feature_dfs.append(feature_df)

    # Concatenate feature DataFrames for each quantity column
    training_df = pd.concat(feature_dfs, axis=1).set_index(id_col)

    for qty_col in qty_list:
        for days in days_lst:
            df_tran_training[trans_date].max()
            window = df_tran_training[trans_date] > (pd.to_datetime(performance_date) - pd.DateOffset(days=days))
            
            feature_name_total = f'{qty_col}_{days}_days_total'
            feature_name_avg = f'{qty_col}_{days}_days_avg'
            feature_name_std = f'{qty_col}_{days}_days_std'

            training_df[feature_name_total] = df_tran_training[window].groupby(id_col)[qty_col].sum()
            training_df[feature_name_avg] = df_tran_training[window].groupby(id_col)[qty_col].mean()
            training_df[feature_name_std] = df_tran_training[window].groupby(id_col)[qty_col].std()
            
    training_df = training_df.fillna(0)
    target_df = target_df.fillna(0)

    return [training_df, target_df]


'''
Example:
    
tran_df = pd.read_excel('retail.xlsx')
tran_df.dtypes
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity']>0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]

df, id_col, trans_date, qty_list, to_buy, \
performance_date, performance_days =\
tran_df.copy(), 'Customer ID', 'InvoiceDate', ['Quantity'], 1, '2010-07-01', 25

len(set(df['Customer ID']))

training_df, target_df = \
train_target_df(df, id_col, trans_date, qty_list, to_buy, 
              performance_date, performance_days, days_lst=[10, 30, 50])

len(set(training_df['Customer ID']))
list(training_df.columns)
training_df.shape
training_df.isnull().sum()

c = ['Recency', 'EarlyDays', 'Frequency', 'Quantity_10_days_total',
     'Quantity_10_days_avg', 'Quantity_10_days_std']
   
training_df[c].reset_index().head(30)
'''

###########################################################

'''
16) Function Description:
The get_XY function is designed to merge training and target data,
 handle missing values for the target variable, and perform feature 
 selection based on correlation. It takes as input DataFrames 
 containing training and target data, a unique identifier column name,
 a correlation threshold, and a list of column names to be removed. 
 The function uses left merge to combine training and target data, 
 filling missing target values with 0. It then removes specified columns,
 performs correlation-based feature selection, and returns X (features) and y
 (target) DataFrames for use in predictive modeling. This function ensures 
 that the data is prepared for modeling by addressing missing values and 
 selecting relevant features.
'''

def get_XY(training_df, target_df, id_col, cor_thresh, rmv_cols):
    # Merge training and target data
    complete_df = pd.merge(training_df, target_df, on=id_col, how='left')
    # Handle missing values for target
    complete_df.fillna(0, inplace=True)
    
    cols = list(complete_df.columns)

    for itv in rmv_cols:
        if itv in cols:
            cols.remove(itv)    
    
    # Feature selection using correlation study
    selected_features = getcorr(complete_df['target'], complete_df, cols, correlation_threshold = cor_thresh)['varname']
        
    # Prepare data for modeling
    X = complete_df[selected_features]
    y = complete_df['target']
    
    return [X, y]

'''
Example:
cor_thresh, rmv_cols = 0.03, ['Customer ID', 'target']
X, y = get_XY(training_df, target_df, id_col, cor_thresh, rmv_cols)

print ('---------columns of X--------')
print (X.dtypes)
print ("--------row and coumn count of X--------")
print (X.shape)
print ("--------Target rate--------")
print (y.mean())


'''

##########################################################

'''
### 
17) Function Purpose:
The `lift_chart` function evaluates a model's performance by creating a lift table based on predicted probabilities (`pred`) and actual outcomes (`actual`). It divides observations into specified deciles and calculates metrics such as target rate, cumulative target rate, lift, and the Kolmogorov-Smirnov (KS) statistic. Optionally, it generates visualizations including a bar chart illustrating target and predicted target rates and a Lorenz curve depicting cumulative target and non-target rates.

### Parameters:
1. `X_test`: DataFrame containing the columns for predicted probabilities (`pred`) and actual outcomes (`actual`).
2. `pred`: Column name representing the predicted probabilities.
3. `actual`: Column name representing the actual binary outcomes.
4. `decn`: Number of deciles to group the data.
5. `plot_flag`: Binary flag (default True) to enable/disable visualizations.

### Additional Plotting:
- If `plot_flag` is True, the function generates a bar chart for target and predicted target rates and a Lorenz curve for cumulative target and non-target rates.

### Output:
The function returns a DataFrame (`score_dec`) containing decile-wise metrics, including target rate, cumulative target rate, lift, and the KS statistic.
'''

def lift_chart(X_test, pred, actual, decn, plot_flag=True):
    
    DATA = X_test.sort_values([pred])
    DATA['rk'] = list(range(1, 1+ len(DATA)))
    parts = len(DATA) / decn
    DATA['grp'] = np.floor((DATA['rk'] / parts))
    DATA['grp'].value_counts()
    DATA['grp'] = DATA.apply(makerkgrp, axis = 1, args=([decn]))
    DATA['cnt'] = 1
    DATA['non_actual'] = 1 - DATA[actual]

    grouped_data = DATA.groupby('grp')
    score_dec = pd.concat([
        grouped_data['cnt'].sum().rename('grp_cnt'),
        grouped_data[pred].mean().rename('pred_target_rate'),
        grouped_data[actual].mean().rename('target_rate'),
        grouped_data[actual].sum().sort_index(ascending=False).cumsum().rename('target_cumrate') / DATA[actual].sum(),
        grouped_data['non_actual'].sum().sort_index(ascending=False).cumsum().rename('nontarget_cumrate') / (len(DATA) - DATA[actual].sum())
    ], axis=1).reset_index()

    score_dec['grp'] = decn - score_dec['grp']
    score_dec = score_dec.sort_values('grp')
    score_dec = score_dec[score_dec['grp']>0]

    avg_target_rate = DATA[actual].mean()
    score_dec['lift'] = (100 * score_dec['target_rate'] / avg_target_rate).round(3)
    score_dec['ks'] = (np.abs(score_dec['target_cumrate'] - score_dec['nontarget_cumrate'])).round(3)

    if plot_flag:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        score_dec[['target_rate', 'pred_target_rate']].plot(kind="bar", ax=axes[0])
        axes[0].set_title("target rate and predicted target rate")
        axes[0].set_xlabel("ranking groups")
        axes[0].set_ylabel("model target rate")

        axes[1].plot(score_dec['target_cumrate'], label='Cumulative Target Rate', linestyle='dashed', marker='o', linewidth=0.7, markersize=3)
        axes[1].plot(score_dec['nontarget_cumrate'], label='Cumulative Non-Target Rate', linestyle='dashed', marker='o', linewidth=0.7, markersize=3)
        axes[1].set_title('Cumulative Target and Non-Target Rate: Lorenz Curve')
        axes[1].legend()

    return score_dec


##############################################################

'''
18) # Function Description:

 The model_fit function builds different binary classifier models depending on the specified algorithm. 
 It takes the input features X, target variable y, test ratio, random state, and model_algorithm (default is 'lgb').
 The function returns a dictionary containing the model name, lift chart results, model performance metrics, and feature importances.

 Parameters:
 - X (pd.DataFrame): The input features.
 - y (pd.Series): The target variable.
 - test_ratio (float): The ratio of the test set.
 - random_st (int): Random state for reproducibility.
 - model_algorithm (str): The algorithm for building the classifier ('lgb', 'xgb', or 'logis').

 Returns:
 dict: A dictionary containing model information, lift chart results, performance metrics, and feature importances.

'''

def model_fit(X, y, test_ratio, random_st, model_algorithm='lgb'):
    
    def ks_stat(y, yhat):
        return ks_2samp(yhat[y==1], yhat[y!=1]).statistic
    
    # Standardizing the input features
    X_STD = (X - X.min()) / (X.max() - X.min())
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_STD, y, 
                    test_size=test_ratio, random_state=random_st)
    
    features = list(X_train.columns)
    
    # Initializing models with default parameters
    lgb_para = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 5,
        'num_leaves': 31,
        'min_child_samples': 20
    }

    xgb_para = {
        'booster': 'gbtree',
        'learning_rate': 0.3,
        'n_estimators': 300,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0.0
    }

    logis_para = {'verbose': 0,
                  'tol': 0.0005,
                  'C': 1.2,
                  'solver': 'lbfgs',
                  'max_iter': 150,
                  'l1_ratio': 0.0001}
    
    if model_algorithm == 'lgb':
        model = lgb.LGBMClassifier(**lgb_para)
    elif model_algorithm == 'xgb':
        model = xgb.XGBClassifier(**xgb_para)
    elif model_algorithm == 'logis':
        model = LogisticRegression(**logis_para)

    # Fitting the model
    model.fit(X_train, y_train)
    # Predicting probabilities on the test set
    y_pred = model.predict_proba(X_test)[:, 1]
    # Calculating AUC score
    auc_score = roc_auc_score(y_test, y_pred)
    # Calculating KS statistic
    ks_statistic = ks_stat(y_test, y_pred)
    # Calculating ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    # Creating ROC data DataFrame
    roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    # Adding predictions and actual values to the test set
    X_test['pred'] = y_pred
    X_test['actual'] = y_test
    
    # Calculating lift chart results
    result_lift = lift_chart(X_test, 'pred', 'actual', 10, plot_flag=True)

    if model_algorithm == 'lgb':
        importance_variable = list(zip(features, model.feature_importances_))
    elif model_algorithm == 'xgb':
        importance_variable = list(zip(features, model.feature_importances_))
    elif model_algorithm == 'logis':
        importance_variable = list(zip(features, abs(model.coef_[0])))
    
    # Sorting feature importances
    importance_variable.sort(key=lambda t: t[1], reverse=True)
    # Creating DataFrame with feature importances
    df_imp = pd.DataFrame(importance_variable, columns=['varname', 'imp'])
    # Creating performance dictionary
    performance = {'AUC': auc_score, 'KS': ks_statistic,
                   'fpr': fpr, 'tpr': tpr, 'roc_data': roc_data}
    
    # Creating result dictionary
    result_dict = {'model': model_algorithm, 'lift': result_lift, 
                   'performance': performance, 'importance': df_imp}
    
    return result_dict


'''
Example: 
test_ratio, random_st, model_algorithm = 0.3, 12, 'lgb'
 
res_model = model_fit(X, y, test_ratio, random_st, model_algorithm)

res_model['lift']

print ('-----AUC---------')
print (res_model['performance']['AUC'])

print ('-----KS---------')
print (res_model['performance']['KS'])

res_model['importance']
'''

