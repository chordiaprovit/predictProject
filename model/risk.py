import numpy as np
np.seterr(divide='ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle

def formatData(data):
 '''drop rows missing pertinent information'''
 data.dropna(subset=["LoanAmount", "Loan_Status"], inplace=True)
 '''assuming that loan term is 360 if not filled in'''
 data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360, inplace=True)
 '''converting string fields into Boolean and assuming na is not married'''
 data.Married = data.Married.map({"Yes": True, "No": False})
 data.Married.fillna(False, inplace=True)
 '''converting dependents from string to int assuming 3+ is 3 and missing is 0'''
 data.Dependents = data.Dependents.map({"0": int(0), "1": int(1), "2": int(2), "3+": int(3)})
 data.Dependents.fillna(int(0), inplace=True)
 '''covert education to Boolean and assuming blanks are False'''
 data.Education = data.Education.map({"Graduate": True, "Not Graduate": False})
 data.Education.fillna(False, inplace=True)
 '''convert male and female to 0 and 1'''
 data.Gender = data.Gender.map({"Male": 0, "Female": 1})
 data.Gender.fillna(0, inplace=True)
 ''' covert self employed to Boolean and assuming blanks are False'''
 data.Self_Employed = data.Self_Employed.map(dict(Yes=True, No=False))
 data.Self_Employed.fillna(False, inplace=True)
 ''' converting Property from string to int'''
 data.Property_Area = data.Property_Area.map({"Urban": 1, "Semiurban": 2, "Rural": 3})
 data.Loan_Status = data.Loan_Status.map({'Y': 1, 'N': 0})
 data.fillna(0, inplace=True)
 return data

def formatDataOHE(data):
 '''drop rows missing pertinent information'''
 # data.dropna(subset=["LoanAmount", "Loan_Status"], inplace=True)
 '''assuming that loan term is 360 if not filled in'''
 data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360, inplace=True)
 '''converting string fields into Boolean and assuming na is not married'''
 data.Married = data.Married.map({"Yes": True, "No": False})
 # data.Married.fillna("No", inplace=True)
 '''converting dependents from string to int assuming 3+ is 3 and missing is 0'''
 data.Dependents = data.Dependents.map({"0": int(0), "1": int(1), "2": int(2), "3+": int(3)})
 # data.Dependents.fillna(int(0), inplace=True)
 '''covert education to Boolean and assuming blanks are False'''
 data.Education = data.Education.map({"Graduate": True, "Not Graduate": False})
 # data.Education.fillna("Not Graduate", inplace=True)
 '''convert male and female to 0 and 1'''
 data.Gender = data.Gender.map({"Male": 0, "Female": 1})
 # data.Gender.fillna("Female", inplace=True)
 ''' covert self employed to Boolean and assuming blanks are False'''
 data.Self_Employed = data.Self_Employed.map(dict(Yes=True, No=False))
 # data.Self_Employed.fillna("No", inplace=True)
 ''' converting Property from string to int'''
 data.Property_Area = data.Property_Area.map({"Urban": 1, "Semiurban": 2, "Rural": 3})
 data.Loan_Status = data.Loan_Status.map({'Y': 1, 'N': 0})
 # data.fillna(0, inplace=True)
 return data

'''Part 1 read data in and manipulate it
read in data'''
data = pd.read_csv('credit_risk.csv')


'''dataset for One Hot Encoding Example'''
dataOHE=data.drop(columns=["Loan_ID"])
'''drop Loan_ID as it is not needed'''
data=data.drop(columns=["Loan_ID"])


'''manipulate data for learning'''
data=formatData(data)
# print("data after preprocessing:\n{}".format(data))

'''Part 2 Split data up
Split into X and Y sets'''
'''In Credit we are not allowed to use Gender or Married so we are dropping them from the dataset'''
X=data.drop(columns=["Loan_Status","Gender","Married"],inplace=False)
Y=data["Loan_Status"]


'''split into train and test (75% train 25% test)'''
'''You can play with the splitting parameters to see if it makes a difference'''
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(X,Y, test_size=0.25, random_state=111)


'''Part 3 train model'''
'''You can play with the Algorithms parameters here view the below site to see all of them, we show the default ones'''
'''https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html'''
treeclassifier = tree.DecisionTreeClassifier()
treeclassifier.fit(train_data_x, train_data_y)
# print("treeclassifier_training_parameters_used:{}".format(treeclassifier.get_params()))

'''Part 4 Test'''
predictions=treeclassifier.predict(test_data_x)
scoretrain=treeclassifier.score(train_data_x,train_data_y)
scoretest=treeclassifier.score(test_data_x,test_data_y)

'''Part 5 review Testing results'''
# print("Predictions")
# print(predictions)
print("Training Score")
print(scoretrain)
print("Test Data Score")
print(scoretest)
print("---------------------------")

'''Part 6 review features and determine if changes would help increase accuracy'''
'''Feature Analysis'''
'''Re-look at assumptions'''
'''drop some columns or add columns'''
# print("columns for feature importance: {}".format(X.keys()))
# print("feature_importance:{}".format(treeclassifier.feature_importances_))

#########################################################################################################################
'''Look to other Algorithms to see if they would help'''
'''Part 6 Logistic Regression'''
from sklearn.linear_model import LogisticRegression
'''Once again you can play with the algorithms paramters here'''
'''https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'''
lr=LogisticRegression(solver='liblinear', class_weight='balanced')
lr.fit(train_data_x, train_data_y)
# print("logistic_regression_training_parameters_used:{}".format(lr.get_params()))
lrpredict=lr.predict(test_data_x)
lrscoretrain=lr.score(train_data_x,train_data_y)
lrscoretest=lr.score(test_data_x,test_data_y)

'''Review Testing results'''
# print("Predictions")
# print(lrpredict)
print("Training Score")
print(lrscoretrain)
print("Test Data Score")
print(lrscoretest)

filename = '../app/lr_model.sav'
pickle.dump(lr, open(filename, 'wb'))
print("model saved")
print("-----------------------------------")



'''Part 7 Logistic Regression with fewer columns'''
'''Here we show how having more or less columns can affect the score'''
'''Generate a smaller feature set'''
smallerX=data.drop(columns=["Loan_Status","Loan_Amount_Term","Credit_History","LoanAmount","CoapplicantIncome","ApplicantIncome"],inplace=False)
smallerY=data["Loan_Status"]
# print("columns of data in smallerX: {}".format(smallerX.keys()))
'''Split it to test and train'''
smalltrain_data_x, smalltest_data_x, smalltrain_data_y, smalltest_data_y = train_test_split(smallerX,smallerY, test_size=0.25, random_state=111)
'''Train it'''
lrsmaller=LogisticRegression(solver='liblinear', class_weight='balanced')
lrsmaller.fit(smalltrain_data_x, smalltrain_data_y)

'''Get the scores to compare'''
smallerlrpredict=lrsmaller.predict(smalltest_data_x)
smallerlrscoretrain=lrsmaller.score(smalltrain_data_x,smalltrain_data_y)
smallerlrscoretest=lrsmaller.score(smalltest_data_x,smalltest_data_y)

# print("Smaller Predictions")
# print(smallerlrpredict)
print("Smaller Training Score")
print(smallerlrscoretrain)
print("Smaller Test Data Score")
print(smallerlrscoretest)
print("------------------------------")
'''Compare the scores'''
'''
print("\n\nCompare scores")
print("Training Score:\nDecision Tree:{} Logistic Regression:{} Smaller Logistic Regression:{}".format(scoretrain,lrscoretrain,smallerlrscoretrain))
print("Testing Score:\nDecision Tree:{} Logistic Regression:{} Smaller Logistic Regression:{}".format(scoretest,lrscoretest,smallerlrscoretest))
'''
'''Part 8 One Hot encoding Example'''
dataOHE = formatDataOHE(dataOHE)
df = pd.get_dummies(dataOHE, columns=[
 "Gender", "Married", "Dependents", "Education", "Self_Employed",
 "Property_Area"])

#OHEX=df.drop(columns=["Loan_Status","Gender_Male","Married_No","Dependents_0","Education_Not Graduate","Property_Area_Urban"],inplace=False)
OHEX=df.drop(columns=["Loan_Status"],inplace=False)
OHEY=df["Loan_Status"]
# print('OHEX keys:\n{}'.format(OHEX.keys()))

OHEtrain_data_x, OHEtest_data_x, OHEtrain_data_y, OHEtest_data_y = train_test_split(OHEX,OHEY, test_size=0.25, random_state=111)
OHElr=LogisticRegression(solver='liblinear', class_weight='balanced')
OHElr.fit(OHEtrain_data_x, OHEtrain_data_y)

# print("columns for feature importance: {}".format(OHEX.keys()))
#print("feature_importance:{}".format(OHElr.))

'''Get the scores to compare'''
OHElrpredict=OHElr.predict(OHEtest_data_x)
OHElrscoretrain=OHElr.score(OHEtrain_data_x,OHEtrain_data_y)
OHElrscoretest=OHElr.score(OHEtest_data_x,OHEtest_data_y)

# print("OHE Predictions")
# print(OHElrpredict)
print("OHE Training Score")
print(OHElrscoretrain)
print("OHE Test Data Score")
print(OHElrscoretest)

# save the model to disk
filename = 'OHElr_model.sav'
pickle.dump(OHElr, open(filename, 'wb'))
print("-------------------------------------")
'''Compare the scores'''
'''
print("\n\nCompare scores")
print("Training Score:\nDecision Tree:{} Logistic Regression:{} Smaller Logistic Regression:{} OHElr: {}".format(scoretrain,lrscoretrain,smallerlrscoretrain,OHElrscoretrain))
print("Testing Score:\nDecision Tree:{} Logistic Regression:{} Smaller Logistic Regression:{} OHElr: {}".format(scoretest,lrscoretest,smallerlrscoretest,OHElrscoretest))
'''''