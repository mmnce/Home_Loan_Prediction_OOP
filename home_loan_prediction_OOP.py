# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:49:30 2022

@author: morin
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class Test_dataset(object):
    
    def __init__(self, name_file : str):
        
        #Home_Loans_Dataset
        self.name_file = name_file
        self.data = pd.read_csv(self.name_file,sep=',')
        
        #y, the target value
        self.y = self.data['Loan_Status']
        
        #X, all features
        self.numerical_columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount',"Loan_Amount_Term"]
        self.qualitative_columns = ['Gender','Married','Credit_History','Education']
        
    def view_global_information_dataset(self):
        data_shape = self.data.shape
        print('There are {} rows and \n {} columns.'.format(data_shape[0],data_shape[1]))
        print(self.data.columns)
        for column in self.data.columns:
            print(column," ==>",type(self.data[column]),"\n",self.data[column].unique())
            
    def view_target_value(self):
        y_values_counts = self.y.value_counts()
        print(y_values_counts)
        sns.barplot(y_values_counts.index,y_values_counts)
        plt.gca().set_ylabel('frequency')
        plt.show()
    
    def view_features_qualitative_columns(self):
        
        fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(20, 15))
        counter = 0
        for column in self.qualitative_columns:
            column_value_counts = self.data[column].value_counts()

            if column == 'Gender':
                
                trace_x = 0
                trace_y = 0
                
            if column == 'Married':
                
                trace_x = 0
                trace_y = 1
            
            if column == 'Credit_History':
                
                trace_x = 1
                trace_y = 0
            
            if column == 'Education':
                
                trace_x = 1
                trace_y = 1
    
            x_pos = np.arange(0, len(column_value_counts))

    
            axs[trace_x, trace_y].bar(x_pos, column_value_counts.values, tick_label = column_value_counts.index)
            axs[trace_x, trace_y].set_title(column)
            #add_value_label( column_value_counts.index)

            for tick in axs[trace_x, trace_y].get_xticklabels():
                tick.set_rotation(90)

                counter += 1

        plt.show()
    
    def view_features_numerical_columns(self):
        
       for column in self.numerical_columns:
            
            if column == "ApplicantIncome":
                plt.figure(1)
                plt.subplot(121)
                sns.distplot(self.data[column]);
                plt.subplot(122)
                self.data[column].plot.box(figsize=(16,5))
                plt.show()
                
            elif column == "CoapplicantIncome":
                plt.figure(1)
                plt.subplot(121)
                sns.distplot(self.data[column]);
                plt.subplot(122)
                self.data[column].plot.box(figsize=(25,10))
                plt.show()
                
            elif column == "LoanAmount":
                plt.figure(1)
                plt.subplot(121)
                sns.distplot(self.data[column].dropna());
                plt.subplot(122)
                self.data[column].plot.box(figsize=(20,20))
                plt.show()
                
            elif column == "Loan_Amount_Term":
                plt.figure(1)
                plt.subplot(121)
                sns.distplot(self.data[column]);
                plt.subplot(122)
                self.data[column].plot.box(figsize=(15,15))
                plt.show()

class Dataset_preparation(object):
    
    def __init__(self, name_file : str):
        
        self.name_file = name_file
        self.data = pd.read_csv(self.name_file,sep=',')
    
    def clean_up_dataset(self):
        
        self.data["Gender"].fillna(self.data["Gender"].mode()[0],inplace=True)
        self.data["Married"].fillna(self.data["Married"].mode()[0],inplace=True)
        self.data["Self_Employed"].fillna(self.data["Self_Employed"].mode()[0],inplace=True)
        self.data["Credit_History"].fillna(self.data["Credit_History"].mode()[0],inplace=True)
        self.data = self.data.dropna(0)
        self.data = self.data.drop(['Loan_ID'],axis=1)
        self.data = self.data.drop(['Dependents'],axis=1)
        
        data_shape = self.data.shape
        data_null = self.data.isnull().sum()

        return print(self.data, data_shape, data_null)
    
    def features_engineering(self):
        
        self.data['Loan_Status'].replace('N', 0,inplace=True)
        self.data['Loan_Status'].replace('Y', 1,inplace=True)

        self.data['Gender'].replace('Male', 1,inplace=True)
        self.data['Gender'].replace('Female', 0,inplace=True)

        self.data['Married'].replace('Yes', 1,inplace=True)
        self.data['Married'].replace('No', 0,inplace=True)

        self.data['Self_Employed'].replace('Yes', 1,inplace=True)
        self.data['Self_Employed'].replace('No', 0,inplace=True)

        self.data['Education'].replace('Graduate', 1,inplace=True)
        self.data['Education'].replace('Not Graduate', 0,inplace=True)

        self.data['Property_Area'].replace('Rural', 1,inplace=True)
        self.data['Property_Area'].replace('Urban', 0,inplace=True)
        
        self.data["TotalIncome"]= self.data["ApplicantIncome"] + self.data["CoapplicantIncome"]
        self.data[["TotalIncome"]].head(2)
        
        self.data["EMI"] = self.data["LoanAmount"] / self.data["Loan_Amount_Term"]
        self.data[["EMI"]].head(4)
        
        self.data["Balance_Income"] = self.data["TotalIncome"] - self.data["EMI"] * 1000 # To make the units equal we multiply with 1000
        self.data[["Balance_Income"]].head(2)
        
        self.data = self.data.drop(['ApplicantIncome'], axis=1)
        self.data = self.data.drop(['CoapplicantIncome'], axis=1)
        self.data = self.data.drop(['LoanAmount'], axis=1)
        self.data = self.data.drop(['Loan_Amount_Term'], axis=1)
        
        data_head = self.data.head()
        
        return print(data_head, self.data, self.data.shape)
    
    def get_prepared_dataset(self):
        
        return self.data

class Test_prediction_model(object):
    
    def __init__(self, prepared_dataset):
        
        self.prepared_dataset = prepared_dataset
        self.data_ready_for_prediction = self.prepared_dataset
        self.y = self.data_ready_for_prediction['Loan_Status']
        self.X = self.data_ready_for_prediction.copy()
    
    def test_model(self):
        
        del self.X['Loan_Status']
        
        self.X = pd.get_dummies(self.X)
        
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.3,random_state=1)
        
        print("X_train","\n", X_train)
        print("X_test","\n", X_test)
        print("y_train", "\n", y_train)
        print("y_test", "\n", y_test)
        
    
        logistic_model = LogisticRegression(random_state=1)
        logistic_model.fit(X_train,y_train)
        
        pred_test_logistic = logistic_model.predict(X_test)
        score_logistic = accuracy_score(pred_test_logistic,y_test)*100
        
        
        print("Accuracy on training set: {:.2f}".format(logistic_model.score(X_train,y_train)))
        print("Accuracy on training set: {:.2f}".format(logistic_model.score(X_test,y_test)))
        print(score_logistic)
        
        return score_logistic

class Prediction_model(object):
    
    def __init__(self, prepared_dataset):
    
        self.prepared_dataset = prepared_dataset
        self.data_ready_for_prediction = self.prepared_dataset
        self.y = self.data_ready_for_prediction['Loan_Status']
        self.X = self.data_ready_for_prediction.copy()
    
    def forest_model(self):
        
        del self.X['Loan_Status']
        
        self.X = pd.get_dummies(self.X)
        
        X_train,X_test,y_train,y_test = train_test_split(self.X, self.y, test_size=0.4, random_state=2)
        
        forest_model = RandomForestClassifier(random_state=1,max_depth=50,n_estimators=10)
        forest_model.fit(X_train,y_train)
        pred_test_forest = forest_model.predict(X_test)
        score_forest = accuracy_score(pred_test_forest,y_test)*100

        test_result=forest_model.score(X_test,y_test)*100

        print(test_result)
        print(score_forest)
        
        return test_result, score_forest
        

data = Test_dataset("Home_Loans_Dataset.csv")
data.view_global_information_dataset()
data.view_target_value()
data.view_features_qualitative_columns()
data.view_features_numerical_columns()

clean_data = Dataset_preparation("Home_Loans_Dataset.csv")
clean_data.clean_up_dataset()
clean_data.features_engineering()
prepared_data = clean_data.get_prepared_dataset()
print(prepared_data)
print(type(prepared_data))

test_model_data = Test_prediction_model(prepared_data)
test_model_data.test_model()

prediction_data = Prediction_model(prepared_data)
prediction_data.forest_model()



