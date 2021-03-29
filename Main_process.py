path = "C:/Users/sijian.xuan/Desktop/NL AIB pythontransfer/python_version"
# path = 'C:/Users/sijian.xuan/Desktop/python_version' # adapt here if needed
input_path = path + "/input"   # adapt here if needed
output_path = path + "/output" # adapt here if needed
package_path = path + '/package' # adapt here if needed

testmnd = "2009" # adapt here

"""
import packages
"""
import os
import warnings
import numpy as np
import pandas as pd
import calendar
import time
import datetime

os.chdir(package_path)
from indication_split_macro import *
os.chdir(path)

warnings.filterwarnings('ignore')

"""
read files
need to import: xbt_01, xbt_01_comed
"""
xbt_01 = pd.read_csv(input_path+"/xbt_01.csv")
# xbt_01_comed = pd.read_csv(path+"/input/xbt_01_comed.csv")
knmp_raw = pd.read_csv(input_path+"/knmp_raw_csv.zip") # adapt here if needed
knmp_univ = pd.read_csv(input_path+"/knmp_2009_csv.zip") # adapt here if needed
input_focus = pd.read_csv(input_path+"/input_focus.zip") # adapt here if needed
input_comed = pd.read_csv(input_path+"/input_comed.zip") # adapt here if needed
pat_indication = pd.read_csv(input_path + "/pat_indication.csv") # adapt here if needed


# globals()["knmp_"+testmnd] = knmp_univ.copy()

# add atc4 infomation into input_focus, or we can do this earlier
# knmp_raw = knmp_raw.drop_duplicates(subset = "fcc")

"""
Model 2 - Split Rheuma, Derma and Gastro indications
"""
start_time = datetime.datetime.now()
out_feature_python = Generic_Feature_Selection(Focus = input_focus, Comed = input_comed,testmnd=testmnd,
                                                pack_id = 'fcc',Focus_Consumption='Y', 
                                                Focus_Line='Y', Focus_Specc='Y',
                                                TOPN=10, Focus_Seasonal='Y', Focus_Regional='Y',
                                                Focus_Insurance='Y', Comed_Consumption='Y', 
                                                Comed_Specc='Y', Comed_Line='N',
                                                Selected_Comed_list='N',cohort_size=30000)
out_feature_python.to_csv(output_path + "/out_feature_python.csv")
# generic_features = pd.read_csv(output_path + "/out_feature_python.csv")
generic_features = out_feature_python.copy()



pat_indication_M2 = get_labels(model_nr=2, inpatindi=pat_indication)
pat_features_M2 = add_comed_features_per_model(in_comed=input_comed,focus_indication=pat_indication_M2,
                                               in_feature=generic_features,model_nr=2,pack_id_type="fcc",
                                               knmp_univ=knmp_univ)

data4model2, data2pred2 = create_data4model_data2pred(features=pat_features_M2, 
                                                      focus_indication=pat_indication_M2,
                                                      model_nr=2, export="N", pydir=output_path)


pat_indication_M3 = get_labels(model_nr=3, inpatindi=pat_indication)
pat_features_M3 = add_comed_features_per_model(in_comed=input_comed,focus_indication=pat_indication_M3,
                                               in_feature=generic_features,model_nr=3,pack_id_type="fcc",
                                               knmp_univ=knmp_univ)
data4model3, data2pred3 = create_data4model_data2pred(features=pat_features_M3, 
                                                      focus_indication=pat_indication_M3,
                                                      model_nr=3, export="N", pydir=output_path)


pat_indication_M4 = get_labels(model_nr=4, inpatindi=pat_indication)
pat_features_M4 = add_comed_features_per_model(in_comed=input_comed,focus_indication=pat_indication_M4,
                                               in_feature=generic_features,model_nr=4,pack_id_type="fcc",
                                               knmp_univ=knmp_univ)
data4model4, data2pred4 = create_data4model_data2pred(features=pat_features_M4, 
                                                      focus_indication=pat_indication_M4,
                                                      model_nr=4, export="N", pydir=output_path)


pat_indication_M5 = get_labels(model_nr=5, inpatindi=pat_indication)
pat_features_M5 = add_comed_features_per_model(in_comed=input_comed,focus_indication=pat_indication_M5,
                                               in_feature=generic_features,model_nr=5,pack_id_type="fcc",
                                               knmp_univ=knmp_univ)
data4model5, data2pred5 = create_data4model_data2pred(features=pat_features_M5, 
                                                      focus_indication=pat_indication_M5,
                                                      model_nr=5, export="N", pydir=output_path)


end_time = datetime.datetime.now()
print("total time used: {}".format(end_time-start_time))


# add more features


pat_indi_firstdax = xbt_01.groupby(["pat","line"])["numdax_number"].min().reset_index()
pat_indi_firstdax2 = pat_indi_firstdax.pivot_table(values = "numdax_number",
                                     index = "pat",
                                     columns = "line", aggfunc = sum,
                                     fill_value = 0)
pat_indi_firstdax2.columns = ["firstdax_" + x for x in pat_indi_firstdax2.columns.tolist()]
print(pat_indi_firstdax2.shape)

def add_features(data=None):
    outfile = pd.merge(data, pat_indi_firstdax2, how="left", on = "pat")
    return outfile
    
data4model2 = add_features(data4model2)
data2pred2 = add_features(data2pred2)
data4model3 = add_features(data4model3)
data2pred3 = add_features(data2pred3)
data4model4 = add_features(data4model4)
data2pred4 = add_features(data2pred4)

# data4model2.to_csv(output_path + "/data4model_M2.csv")
# data2pred2.to_csv(output_path + "/data2pred_M2.csv")
# data4model3.to_csv(output_path + "/data4model_M3.csv")
# data2pred3.to_csv(output_path + "/data2pred_M3.csv")
# data4model4.to_csv(output_path + "/data4model_M4.csv")
# data2pred4.to_csv(output_path + "/data2pred_M4.csv")
# data4model5.to_csv(output_path + "/data4model_M5.csv")
# data2pred5.to_csv(output_path + "/data2pred_M5.csv")
"""
data4model2 = pd.read_csv(output_path + "/data4model_M2.csv")
data2pred2 = pd.read_csv(output_path + "/data2pred_M2.csv")
data4model3 = pd.read_csv(output_path + "/data4model_M3.csv")
data2pred3 = pd.read_csv(output_path + "/data2pred_M3.csv")
data4model4 = pd.read_csv(output_path + "/data4model_M4.csv")
data2pred4 = pd.read_csv(output_path + "/data2pred_M4.csv")
data4model5 = pd.read_csv(output_path + "/data4model_M5.csv")
data2pred5 = pd.read_csv(output_path + "/data2pred_M5.csv")
"""
##########################################################################
"""modeling!!"""
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import ensemble, linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from imblearn.under_sampling import RandomUnderSampler
import sys
import seaborn as sns
# Load user-defined packages
os.chdir(package_path)
from modeling_tools import *
# Set working directory - adapt here
os.chdir(path)
os.getcwd()

# pd.set_option('display.max_rows',None) # do not print max_rows in pandas
# np.set_printoptions(threshold =sys.maxsize) # do not print ... in numpy
# warnings.filterwarnings("ignore") # do not print warnings

"""
model2
"""
try:
    data4model2.drop("Unnamed: 0",axis=1,inplace=True)
    data2pred2.drop("Unnamed: 0",axis=1,inplace=True)
except:
    pass

vc_good(data4model2,"indication")

#         counts       %
# Rheuma   16165  65.07%
# Derma     4242  17.07%
# Gastro    3770  15.17%
# Kinder     449   1.81%
# Ogen       218   0.88%

x_data = data4model2.drop(['indication','pat'],axis = 1)
y_data = pd.DataFrame(data4model2['indication']).squeeze()

x_data = pd.get_dummies(x_data)        
x_data = x_data.fillna(0)
column_name = x_data.columns
x_data = pd.DataFrame(Normalizer(norm='l2').transform(x_data))
x_data.columns = column_name  

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.25, stratify=y_data, random_state = 1)
y_train = y_train.squeeze()
y_test = y_test.squeeze()
print("X_train shape: {}\nX_test shape: {}\ny_train shape: {}\ny_test shape: {} ".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# ### Fit the model
# Fit the best model based on tuned parameters
best_clf = ensemble.GradientBoostingClassifier(learning_rate = 0.05, max_depth = 3, 
                                               n_estimators = 100, random_state=42)## select model

# Fit the model and check ConfusionMatrix
best_clf.fit(X_train,y_train)

# Check R-Style confusionMatrix
y_pred = best_clf.predict(X_test).tolist()## change type: object to list, cannot create Confusion Matrix if not change
confusionMatrix(y_pred, y_test).show()## Show the Confusion Matrix
# Classification Report
print('Classification Report:\n',classification_report(y_test,y_pred, target_names=["Derma","Gastro","Kinder","Ogen","Rheuma"]))

#           Reference
# Prediction
#               Derma  Gastro  Kinder  Ogen  Rheuma
#       Derma     988      13       3     0      56
#       Gastro     11     850       2     3      45
#       Kinder      0       3      50     5      10
#       Ogen        0       0       3    29       4
#       Rheuma     62      76      54    18    3926 

# Overall Statistics

#               Accuracy :  0.9408
#    No Information Rate :  0.6506 

# Statistics by Class:

#               Derma    Gastro    Kinder      Ogen    Rheuma
# Precision  0.932075  0.933041  0.735294  0.805556  0.949226
# Recall     0.931197  0.902335  0.446429  0.527273  0.971542

# Classification Report:
#                precision    recall  f1-score   support

#        Derma       0.93      0.93      0.93      1061
#       Gastro       0.93      0.90      0.92       942
#       Kinder       0.74      0.45      0.56       112
#         Ogen       0.81      0.53      0.64        55
#       Rheuma       0.95      0.97      0.96      4041

#     accuracy                           0.94      6211
#    macro avg       0.87      0.76      0.80      6211
# weighted avg       0.94      0.94      0.94      6211

# ### Make prediction
# Data to be predicted
data_predict = data2pred2.drop(['pat','indication'], axis = 1)
data_predict = pd.get_dummies(data_predict)        
data_predict = data_predict.fillna(0)
column_name = data_predict.columns
data_predict = pd.DataFrame(Normalizer(norm='l2').transform(data_predict))
data_predict.columns = column_name  

# Final prediction dataset needs to have the same contents as the training and testing set
pred_final_model = pd.DataFrame(best_clf.predict(data_predict))## predicted indications
pred_final_prob = pd.DataFrame(best_clf.predict_proba(data_predict)) ## predicted probabilities

# Plot top features
feature_importances = pd.concat([pd.DataFrame(x_data.columns), pd.DataFrame(best_clf.feature_importances_)], axis = 1)
feature_importances.columns = ['features', 'importance']
feature_importances.sort_values(by = ['importance'], ascending = False, inplace = True)
#plt.title('Feature importance')
sns.barplot(x='importance', y='features', data=feature_importances.head(), color="b")

# Create a patient list with indication and probability of every indication
final_results = pd.concat([data2pred2['pat'], pd.DataFrame(pred_final_model), pd.DataFrame(pred_final_prob)], axis = 1)
final_results.columns = ['pat','indication','prob_Derma','prob_Gastro','prob_Kinder','prob_Ogen','prob_Rheuma'] # adpat here
final_results.reset_index(drop = True,inplace=True)
final_results.head()

# Check frequency tables
vc_good(final_results,"indication")
#         counts       %
# Rheuma   35376  72.42%
# Gastro    7144  14.63%
# Derma     5405  11.07%
# Kinder     656   1.34%
# Ogen       265   0.54%


compare2 = pd.read_csv(r"C:\Users\sijian.xuan\Desktop\prediction_model2.csv")
vc_good(compare2,"indication")
#         counts       %
# Rheuma   35006  71.65%
# Gastro    7441  15.23%
# Derma     5498  11.25%
# Kinder     713   1.46%
# Ogen       201   0.41%

pd.merge(final_results, compare2, how="outer", on="pat").groupby(["indication_x","indication_y"]).size()
# indication_x  indication_y
# Derma         Derma            5186
#               Gastro             50
#               Kinder              5
#               Rheuma            162
# Gastro        Derma              24
#               Gastro           6943
#               Kinder             11
#               Ogen               10
#               Rheuma            134
# Kinder        Derma               5
#               Gastro             13
#               Kinder            486
#               Ogen               13
#               Rheuma            139
# Ogen          Derma               1
#               Gastro             31
#               Kinder             17
#               Ogen              124
#               Rheuma             92
# Rheuma        Derma             269
#               Gastro            383
#               Kinder            194
#               Ogen               54
#               Rheuma          34425






"""
model3
"""

# Load input data for model and prediction
data4model = data4modeling3.copy() 
data2pred = data2pred3.copy()



# do data_visualization
data_visualization_model3(data4model)
######################################################
x_data = data4model.drop(['indication','pat'],axis = 1)
y_data = pd.DataFrame(data4model['indication'])
y_data_series = y_data.squeeze()

# Check frequency tables
print('Total counts per indication in the dataset: ')
print(y_data_series.value_counts())
print(' ')
print('Market share per indication in the dataset: ')
print(y_data_series.value_counts()/y_data_series.count())


# ### Data Preprocesssing
# **preprocess the whole dataset**
#  1. get dummies for on X_data
#  2. fill nan with 0 on X_data
#  3. normalizer_l2 on X_data
#  4. label encoding on y_data
#  5. use different model to train set
#  6. get accuracy on tst set
pipeline = Pipeline([('dummies,fillna,normalizer', processing()), ])
x_data_copy = x_data.copy()
x_data_pipeline = pd.DataFrame(pipeline.fit_transform(x_data_copy))

pipeline2 = Pipeline([('labelrecoding', label_encoder())])
y_data_copy = y_data.copy()
# x_data_pipeline,y_data_copy are data for modeling


# Split training set and testing sst
X = x_data_pipeline#####Select feature columns
y = y_data_copy#####Select label columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state = 1)
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# ### Fit the model
# Fit the best model based on tuned parameters
GBM_clf = ensemble.GradientBoostingClassifier(learning_rate = 0.05, max_depth = 3, n_estimators = 100)
best_clf = BalancedBaggingClassifier(base_estimator=GBM_clf,ratio='auto',replacement=False,random_state=0)

# Fit the model and check ConfusionMatrix
best_clf.fit(X_train,y_train)

# Check R-Style confusionMatrix

y_pred = best_clf.predict(X_test).tolist()## change type: object to list, cannot create Confusion Matrix if not change
confusionMatrix(y_pred, y_test).show()## Show the Confusion Matrix
# Classification Report
print('Classification Report:\n',classification_report(y_test,y_pred, target_names=["AS","PsA","RA"]))


### prepare input for ROC
n_classes = len(y_train.unique()) # number of indications, if 2 then n_class=1, if >2 then the number of indications
y_score = best_clf.fit(X_train, y_train).decision_function(X_test)
y_test2 = pd.get_dummies(y_test)

ROC(n_classes, y_score, y_test2)
PRC(n_classes,y_test2,y_score)
AUC_model3(best_clf, X_train, y_train, X_test, y_test, n_classes)                 


# ### Make prediction
# Data to be predicted
data_predict = data2pred.drop(['pat','indication'], axis = 1)
data_predict_index = data_predict.index
data_predict_pipeline = pd.DataFrame(pipeline.fit_transform(data_predict))
print(data_predict_pipeline.shape)
data_predict_pipeline.index = data_predict_index
data_predict_pipeline.head()

# Final prediction dataset needs to have the same contents as the training and testing set
pred_final_model = pd.DataFrame(best_clf.predict(data_predict_pipeline))## predicted indications
pred_final_prob = pd.DataFrame(best_clf.predict_proba(data_predict_pipeline)) ## predicted probabilities
pred_final_model.index = data_predict_index
pred_final_prob.index = data_predict_index
# Plot top features (use RandomUnderSampler which maybe a bit different from BalancedBaggingClassifier)
rus = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)

# Adapt X_train, y_train
X_train2 = X_resampled.copy()
y_train2 = y_resampled.copy()

GBM_clf.fit(X_train2, y_train2) 

# Plot top features
feature_importances = pd.concat([pd.DataFrame(x_data_pipeline.columns), pd.DataFrame(GBM_clf.feature_importances_)], axis = 1)
feature_importances.columns = ['features', 'importance']
feature_importances.sort_values(by = ['importance'], ascending = False, inplace = True)
sns.barplot(x='importance', y='features', data=feature_importances.head(), color="b")
#plt.title('Feature importance')

# Create a patient list with indication and probability of every indication
final_results = pd.concat([data2pred['pat'], pd.DataFrame(pred_final_model), pd.DataFrame(pred_final_prob)], axis = 1)
final_results.columns = ['pat','indication','prob_AS','prob_PsA','prob_RA'] # adpat here
final_results.reset_index(drop = True,inplace=True)
final_results.head()

# Check frequency tables
print('Total counts per indication in the dataset: ')
print(final_results['indication'].value_counts())
print(' ')
print('Market share per indication in the dataset: ')
print(final_results['indication'].value_counts()/final_results['indication'].count())

final_results.to_csv("./output/prediction_model3.csv", index = False) # adpat here


"""
model4
"""
data4model = data4modeling4.copy()
data2pred =data2pred4.copy()



# do data_visualization
data_visualization_model4(data4model)
######################################################
x_data = data4model.drop(['indication','pat'],axis = 1)
y_data = pd.DataFrame(data4model['indication'])
y_data_series = y_data.squeeze()


# Check frequency tables
print('Total counts per indication in the dataset: ')
print(y_data_series.value_counts())
print(' ')
print('Market share per indication in the dataset: ')
print(y_data_series.value_counts()/y_data_series.count())


# ### Data Preprocesssing
# **preprocess the whole dataset**
#  1. get dummies for on X_data
#  2. fill nan with 0 on X_data
#  3. normalizer_l2 on X_data
#  4. label encoding on y_data
#  5. use different model to train set
#  6. get accuracy on tst set

pipeline = Pipeline([('dummies,fillna,normalizer', processing()), ])
x_data_copy = x_data.copy()
x_data_pipeline = pd.DataFrame(pipeline.fit_transform(x_data_copy))

pipeline2 = Pipeline([('labelrecoding', label_encoder())])
y_data_copy = y_data.copy()
# x_data_pipeline,y_data_copy are data for modeling


# Split training set and testing sst
X = x_data_pipeline#####Select feature columns
y = y_data_copy#####Select label columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state = 1)
y_train = y_train.squeeze()
y_test = y_test.squeeze()


# ### Fit the model

# Fit the best model based on tuned parameters
GBM_clf = ensemble.GradientBoostingClassifier(learning_rate = 0.1, max_depth = 2, n_estimators = 15)
best_clf = BalancedBaggingClassifier(base_estimator=GBM_clf,ratio='auto',replacement=False,random_state=0)
# Fit the model and check ConfusionMatrix
best_clf.fit(X_train,y_train)

# Check R-Style confusionMatrix
y_pred = best_clf.predict(X_test).tolist()## change type: object to list, cannot create Confusion Matrix if not change

confusionMatrix(y_pred, y_test).show()## Show the Confusion Matrix
# Classification Report
print('Classification Report:\n',classification_report(y_test,y_pred, target_names=["CD","UC"]))


### prepare input for ROC
n_classes = 1 # number of indications, if 2 then n_class=1, if >2 then the number of indications
y_score = best_clf.fit(X_train, y_train).decision_function(X_test).reshape((len(y_test),1)) # score for ROC
y_test2 = (y_test == 'UC').reshape((len(y_test),1))*1

ROC(n_classes, y_score, y_test2)
PRC(n_classes,y_test2,y_score)

# ### Make prediction
# Data to be predicted
data_predict = data2pred.drop(['pat','indication'], axis = 1)
data_predict_index = data_predict.index
data_predict_pipeline = pd.DataFrame(pipeline.fit_transform(data_predict))
print(data_predict_pipeline.shape)
data_predict_pipeline.index = data_predict_index
data_predict_pipeline.head()

# Final prediction dataset needs to have the same contents as the training and testing set
pred_final_model = pd.DataFrame(best_clf.predict(data_predict_pipeline))## predicted indications
pred_final_prob = pd.DataFrame(best_clf.predict_proba(data_predict_pipeline)) ## predicted probabilities
pred_final_model.index = data_predict_index
pred_final_prob.index = data_predict_index

# Plot top features (use RandomUnderSampler which maybe a bit different from BalancedBaggingClassifier)
rus = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)

# Adapt X_train, y_train
X_train2 = X_resampled.copy()
y_train2 = y_resampled.copy()

GBM_clf.fit(X_train2, y_train2) 

# Plot top features
feature_importances = pd.concat([pd.DataFrame(x_data_pipeline.columns), pd.DataFrame(GBM_clf.feature_importances_)], axis = 1)
feature_importances.columns = ['features', 'importance']
feature_importances.sort_values(by = ['importance'], ascending = False, inplace = True)
sns.barplot(x='importance', y='features', data=feature_importances.head(), color="b")
#plt.title('Feature importance')


# Create a patient list with indication and probability of every indication
final_results = pd.concat([data2pred['pat'], pd.DataFrame(pred_final_model), pd.DataFrame(pred_final_prob)], axis = 1)
final_results.columns = ['pat','indication','prob_CD','prob_UC']
final_results.reset_index(drop = True,inplace=True)
final_results.head()

# Check frequency tables
print('Total counts per indication in the dataset: ')
print(final_results['indication'].value_counts())
print(' ')
print('Market share per indication in the dataset: ')
print(final_results['indication'].value_counts()/final_results['indication'].count())


final_results.to_csv("./output/prediction_model4.csv", index = False) # adpat here


"""
put results together
"""
os.chdir(dir_macro)
os.getcwd()

from final_indication01_macro import *
from load_model_csv import *
from proc_freq import *
from validation_b_o_rules import *
from stabilizer import *

os.chdir(path)
os.getcwd()


prediction_model2 = pd.read_csv(path+"/output/prediction_model2.csv")
prediction_model3_bbc = pd.read_csv(path + "/output/prediction_model3.csv")
prediction_model4_bbc = pd.read_csv(path + "/output/prediction_model4.csv")


pat2addindication = input_focus['pat'].sort_values().unique()
final_indication_01 = final_indication_01(pat_indication_nodup,pat_indication_total, prediction_model2,prediction_model3_bbc, prediction_model4_bbc,pat2addindication)


"""
also needs to fintune to ensure the final share per indication is compirable to Claims
or Sample datasets
"""
Proc_Freq(final_indication_01,'final_indication')
Proc_Freq(final_indication_01,'indication')

final_indication_02 = final_indication_01[['pat','final_indication']]
final_indication_02 = final_indication_02.rename(index = str, columns = {"final_indication":"indication"})

##create indication_with_features
temp, = np.where((GenericFeatures2.columns.str.startswith('FL_')== True)|(GenericFeatures2.columns.str.startswith('Focus')== True)|(GenericFeatures2.columns == 'pat'))
temp = temp.tolist()

merge1 = pd.merge(GenericFeatures2.iloc[:,temp],final_indication_02,how = "outer")
merge2 = pd.merge(merge1,prediction_model3_bbc,how = "outer")

indication_with_features = merge2.copy()

Val_01 = Validation_based_on_Rules(indication_with_features)

proc_freq(Val_01,"same")
proc_freq_2(Val_01)

final_indication_02.to_csv(path + "/datasets/final_indication_02_" +testmnd +".csv")
final_indication_02 = final_indication_02.rename(index = str, columns = {"indication":"indication_current"})
##这里需要改动 prefinal是用之前的做出来的 这里为了方便先导入了
prevfinal = pd.read_csv(path + "/datasets/final_indication_02_1811.csv")
prevfinal = prevfinal.rename(index = str, columns = {"indication":"indication_previous"})

indication_compare = pd.merge(final_indication_02,prevfinal,how = "outer")
indication_compare.to_csv(path + "/output/indication_compare.csv")


stabilizer04 = stabilizer("201804NEW2")
stabilizer05 = stabilizer("201805NEW2")
stabilizer06 = stabilizer("201806NEW")
stabilizer07 = stabilizer("201807NEW")
stabilizer08 = stabilizer("201808NEW")
stabilizer09 = stabilizer("201809NEW")
stabilizer10 = stabilizer("201810NEW")
stabilizer11 = stabilizer("201811NEW")
stabilizer12 = stabilizer("201812NEW")


dfs = [stabilizer04,stabilizer05,stabilizer06,stabilizer07,stabilizer08]
stabilizer_01 = pd.concat(dfs, ignore_index = True)

stabilizer_01 = stabilizer_01.sort_values(by = ['pat', 'maand'], ascending=[True, False])

### get stabilizer in line 112-123
stabilizer_02 = stabilizer_01.copy().reset_index()

## a trick to encode the nr
time = stabilizer_02['maand'].max()
first_day = datetime.date(time.year, time.month, 1)#first day of this month
days_num = calendar.monthrange(first_day.year, first_day.month)[1] #获取一个月有多少天
first_day_of_next_month = first_day + datetime.timedelta(days = days_num) #当月的最后一天只需要days_num-1即可
first_day_of_next_month = datetime.datetime.combine(first_day_of_next_month, datetime.datetime.min.time())

## get data from most recent 12 months
stabilizer_02['nr'] =  (stabilizer_02['maand'] - first_day_of_next_month)/30 *(-1)
stabilizer_02['nr_2'] = stabilizer_02['nr'].dt.days
stabilizer_02 = stabilizer_02[stabilizer_02['nr_2'] <= 12]
"""every time we need to check line 122-123 whether column 'nr_2' become 1,2,3,4,... no repetition"""

stabilizer_03 = stabilizer_02[['pat','indication']]
stabilizer_03 = pd.DataFrame(stabilizer_03['pat'].groupby([stabilizer_03['indication']]).value_counts())

stabilizer_04 = stabilizer_03.copy()
stabilizer_04.columns = ['count']
stabilizer_04 = stabilizer_04.reset_index()
stabilizer_04 = stabilizer_04.rename(index = str, columns = {"indication":"indication_mode"})

#stabilizer_04.index = stabilizer_04['pat']
temp = stabilizer_04.groupby(['pat']).sum().reset_index()
temp = temp.rename(index = str, columns = {"count":"count_sum"})

stabilizer_04 = pd.merge(stabilizer_04, temp, on = 'pat')
stabilizer_04['perc'] = stabilizer_04['count']/stabilizer_04['count_sum']

stabilizer_04 = stabilizer_04.sort_values(by = ['pat', 'perc'], ascending = [True, False])
stabilizer_04.drop_duplicates(subset='pat', keep='first', inplace = True)

## get latest indication
stabilizer_05 = stabilizer_02.copy()
stabilizer_05 = stabilizer_05[['pat','indication']]
stabilizer_05 = stabilizer_05.rename(index = str, columns = {"indication":"indication_latest"})
#len(stabilizer_05['pat'].unique()) 57467
stabilizer_05.drop_duplicates(subset='pat', keep='first', inplace = True)


# check
#stabilizer_04['pat'].tolist() == stabilizer_05['pat'].tolist()

stabilizer_final = pd.merge(stabilizer_04, stabilizer_05, how = 'outer', on = 'pat')
stabilizer_final['indication'] = np.nan

cond, = np.where(stabilizer_final['perc']>0.5)
cond = cond.tolist()
stabilizer_final['indication'][cond] = stabilizer_final['indication_mode'][cond]

cond2, = np.where(stabilizer_final['perc']<= 0.5)
cond2 = cond2.tolist()
stabilizer_final['indication'][cond2] = stabilizer_final['indication_latest'][cond2]
stabilizer_final = stabilizer_final[['pat','indication']]

#final_indication_03 = stabilizer_final.copy()
stabilizer_final.to_csv(path + "/output/final_indication_"+ testmnd +".(stabilized).csv")



stabilizer_final_temp = stabilizer_final.rename(index = str, columns = {"indication":"indication_current"})
prevfinal_temp = prevfinal.rename(index = str, columns = {"indication":"indication_previous"})
indication_compare2 = pd.merge(stabilizer_final_temp, prevfinal, on = 'pat')

indication_compare2.to_csv(path + "/output/indication_compare(stabilized).csv")





