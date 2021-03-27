# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:37:22 2020

@author: Bonie, Joshua
@author: Paramanamtham, Thileepan
@author: Raman, Rahul
@author: Raturi, Abhishek
@author: Roa, Koolait

"""
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

"--------------------------------------------------"
# 1. Data Exploration
# a complete review and analysis of the dataset including:
"--------------------------------------------------"

##################################################
# a. Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as aproppriate. - use pandas, numpy and any other python packages.

### ----- Load Data ----- ###

#path = "C:/Users/Dhani/Desktop/HCIS/GroupProject/HCIS_GroupProject"
path = "C:/COMP309-003 - Project 02"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
dataframe_theft_bike = pd.read_csv(fullpath,sep=',')

### Set Display - Columns | Rows | Width
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 100)

### ----- Describe Data ----- ###
### Data Set - Records - First 5 | First & Last 5
print("\n\nData Set - Records - First 5")
print(dataframe_theft_bike.head(5))
print("\n\nData Set - Records - First & Last 5")
print(dataframe_theft_bike)

### Data Set - Column - Names | Values | Type | Info | Description/Summaries
print("\n\nData Set - Columns")
print(dataframe_theft_bike.columns.values)
print("\n\nData Set - Data Types")
print(dataframe_theft_bike.dtypes)
print("\n\nData Set - Info")
print(dataframe_theft_bike.info())
print("\n\nData Set - Description")
dataframe_theft_bike.describe()

### Data Set - Shape - (No. of Rows, No. of Columns)
print("\n\nData Set - Shape")
print(dataframe_theft_bike.shape)

### Data Set - Categories
list_categories_dataset = []
for col, col_type in dataframe_theft_bike.dtypes.iteritems():
     if col_type == 'O':
          list_categories_dataset.append(col)
     else:
          dataframe_theft_bike[col].fillna(0, inplace=True)
print("\n\nData Set - Categories")
print(list_categories_dataset)

### Data Set - Unique
print("\n\nData Set - Unique - Primary_Offence")
print(dataframe_theft_bike['Primary_Offence'].unique())

print("\n\nData Set - Unique - Division")
print(dataframe_theft_bike['Division'].unique())

print("\n\nData Set - Unique - Location_Type")
print(dataframe_theft_bike['Location_Type'].unique())

print("\n\nData Set - Unique - Premise_Type")
print(dataframe_theft_bike['Premise_Type'].unique())

print("\n\nData Set - Unique - Bike_Make")
print(dataframe_theft_bike['Bike_Make'].unique())

print("\n\nData Set - Unique - Bike_Model")
print(dataframe_theft_bike['Bike_Model'].unique())

print("\n\nData Set - Unique - Bike_Type")
print(dataframe_theft_bike['Bike_Type'].unique())

print("\n\nData Set - Unique - Bike_Speed")
print(dataframe_theft_bike['Bike_Speed'].unique())

print("\n\nData Set - Unique - Bike_Colour")
print(dataframe_theft_bike['Bike_Colour'].unique())

print("\n\nData Set - Unique - Cost_of_Bike")
print(dataframe_theft_bike['Cost_of_Bike'].unique())

print("\n\nData Set - Unique - Status")
print(dataframe_theft_bike['Status'].unique())

print("\n\nData Set - Unique - Hood_ID")
print(dataframe_theft_bike['Hood_ID'].unique())

print("\n\nData Set - Unique - Neighbourhood")
print(dataframe_theft_bike['Neighbourhood'].unique())

##################################################
# b. Statistical assessments including means, averages, correlations

### ----- Data Set - Statistics/Summaries ----- ###
print(dataframe_theft_bike.describe())

### ----- Data Set - Mean/Average ----- ###
print("\n\nData Set - Mean")
print(np.mean(dataframe_theft_bike))
print("\n\nData Set - Mean - Group By Status")
print(dataframe_theft_bike.groupby('Status').mean())
print("\n\nData Set - Mean - Group By Primary_Offence")
print(dataframe_theft_bike.groupby('Primary_Offence').mean())
print("\n\nData Set - Mean - Cost_of_Bike")
print(np.nanmean(dataframe_theft_bike['Cost_of_Bike']))

def statistics():
    ### ----- Data Set - Correlation ----- ###
    ##### https://pandas.pydata.org/pandas-docs/version/0.25.3/reference/api/pandas.Series.corr.html
    ##### https://stackoverflow.com/questions/48873233/is-there-a-way-to-get-correlation-with-string-data-and-a-numerical-value-in-pand
    ##### https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables-pandas
    def histogram_intersection(a, b):
        v = np.minimum(a, b).sum().round(decimals=1)
        return v
    ### Data Set - Features - Correlation
    print(dataframe_theft_bike.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
    #dataframe_correlation = dataframe_scaled_theft_bike_model_ohe_downsampled.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
    ### Data Set - Features - Correlation - Location_Type | Status
    print(dataframe_theft_bike[['Location_Type', 'Status']].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
    ### Data Set - Features - Correlation - Premise_Type | Status
    print(dataframe_theft_bike[['Premise_Type', 'Status']].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
    ### Data Set - Features - Correlation - Hood_ID | Status
    print(dataframe_theft_bike[['Hood_ID', 'Status']].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
    ### Data Set - Features - Correlation - Bike_Type | Status
    print(dataframe_theft_bike[['Bike_Type', 'Status']].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
    ### Data Set - Features - Correlation - Cost_of_Bike | Status
    print(dataframe_theft_bike[['Cost_of_Bike', 'Status']].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1))
    
    ##################################################
    # c. Missing data evaluations - use pandas, numpy and any other python packages
    
    print(dataframe_theft_bike.isnull().sum())
    
    ##################################################
    # d. Graphs and visualizations - use pandas, matplotlib, seaborn, numpy and any other python packages, you also can use power BI desktop.
    
    ##### https://stackoverflow.com/questions/9651092/my-matplotlib-pyplot-legend-is-being-cut-off/42303455
    ### Bar Chart - Occurences of Theft - Month | Year 
    pd.crosstab(dataframe_theft_bike.Occurrence_Year,dataframe_theft_bike.Occurrence_Month).plot(kind='bar')
    plt.title('Occurences of Theft - Month | Year')
    plt.xlabel('Year')
    plt.ylabel('Occurences')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 01.png', bbox_inches='tight')
    
    ### Stacked Bar Chart - Occurences of Theft | Recovery - Premise_Type | Location_Type
    table=pd.crosstab(dataframe_theft_bike.Premise_Type,dataframe_theft_bike.Location_Type)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Occurences of Theft | Recovery - Premise_Type | Location_Type')
    plt.xlabel('Premise_Type')
    plt.ylabel('Location_Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 02.png', bbox_inches='tight')
    
    ### Bar Chart - Occurences of Theft - Premise_Type | Status
    pd.crosstab(dataframe_theft_bike.Status,dataframe_theft_bike.Premise_Type).plot(kind='bar')
    plt.title('Occurences of Theft | Recovery - Premise_Type | Status')
    plt.xlabel('Status')
    plt.ylabel('Occurences')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 03.png', bbox_inches='tight')
    
    ### Stacked Bar Chart - Occurences of Theft | Recovery - Premise_Type | Status
    table=pd.crosstab(dataframe_theft_bike.Status,dataframe_theft_bike.Premise_Type)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Occurences of Theft | Recovery - Premise_Type | Status')
    plt.xlabel('Status')
    plt.ylabel('Premise_Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 04.png', bbox_inches='tight')
    
    ### Stacked Bar Chart - Occurences of Theft | Recovery - Location_Type | Status
    table=pd.crosstab(dataframe_theft_bike.Status,dataframe_theft_bike.Location_Type)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Occurences of Theft | Recovery - Location_Type | Status')
    plt.xlabel('Status')
    plt.ylabel('Location_Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 05.png', bbox_inches='tight')
    
    ### Stacked Bar Chart - Occurences of Theft | Recovery - Bike_Type | Status
    table=pd.crosstab(dataframe_theft_bike.Status,dataframe_theft_bike.Bike_Type)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
    plt.title('Occurences of Theft | Recovery - Bike_Type | Status')
    plt.xlabel('Status')
    plt.ylabel('Bike_Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 06.png', bbox_inches='tight')
    
    ### Histogram - Occurences of Theft - Premise_Type
    dataframe_theft_bike.Premise_Type.hist()
    plt.title('Occurences of Theft - Cost_of_Bike')
    plt.xlabel('Premise_Type')
    plt.ylabel('Occurences')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 07.png', bbox_inches='tight')
    
    ### Histogram - Occurences of Theft - Bike_Type
    dataframe_theft_bike.Bike_Type.hist()
    plt.title('Occurences of Theft - Bike_Type')
    plt.xlabel('Bike_Type')
    plt.ylabel('Occurences')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(path + '/COMP309-003 - Project 02 - Graph 08.png', bbox_inches='tight')
    
    ### Scatter Plot - Occurences of Theft - Lat | Long
    dataframe_theft_bike.plot(kind='scatter',x='Long',y='Lat').figure.savefig(path + '/COMP309-003 - Project 02 - Graph 09.png', bbox_inches='tight')
    
    plt.show()
    return

"--------------------------------------------------"
# 2. Data Modelling
"--------------------------------------------------"

##################################################
# a. Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.

### ----- Data Set - Data Transformation ----- ###

###  Data Set - Column - Location_Type - Commercial - Hotel
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Commercial Dwelling Unit (Hotel, Motel, B & B, Short Term Rental)', 'Commercial - Hotel', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Commercial - Corporate
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Commercial / Corporate Places (For Profit, Warehouse, Corp. Bldg', 'Commercial - Corporate', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Commercial - Retail
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Convenience Stores', 'Commercial - Retail', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Bar / Restaurant', 'Commercial - Retail', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Dealership (Car, Motorcycle, Marine, Trailer, Etc.)', 'Commercial - Retail', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Commercial - Special-Purpose
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Gas Station (Self, Full, Attached Convenience)', 'Commercial - Special-Purpose', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Industrial
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Construction Site (Warehouse, Trailer, Shed)', 'Industrial', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Public Open Area
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Streets, Roads, Highways (Bicycle Path, Private Road)', 'Public Open Area', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Parking Lots (Apt., Commercial Or Non-Commercial)', 'Public Open Area', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Open Areas (Lakes, Parks, Rivers)', 'Public Open Area', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Residential - Small Housing
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Private Property (Pool, Shed, Detached Garage)', 'Residential - Small Housing', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Private Property Structure (Pool, Shed, Detached Garage)', 'Residential - Small Housing', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Single Home, House (Attach Garage, Cottage, Mobile)', 'Residential - Small Housing', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Residential - Group Housing
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Homeless Shelter / Mission', 'Residential - Group Housing', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Apartment (Rooming House, Condo)', 'Residential - Group Housing', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Retirement / Nursing Homes', 'Residential - Group Housing', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Retirement Home', 'Residential - Group Housing', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Educational
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Schools During Un-Supervised Activity', 'Specialty - Educational', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Schools During Supervised Activity', 'Specialty - Educational', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Universities / Colleges', 'Specialty - Educational', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Financial
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Bank And Other Financial Institutions (Money Mart, Tsx)', 'Specialty - Financial', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Medical
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Hospital / Institutions / Medical Facilities (Clinic, Dentist, Morgue)', 'Specialty - Medical', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Pharmacy', 'Specialty - Medical', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Government
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Police / Courts (Parole Board, Probation Office)', 'Specialty - Government', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Group Homes (Non-Profit, Halfway House, Social Agency)', 'Specialty - Government', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Jails / Detention Centres', 'Specialty - Government', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Other Non-Profit
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == "Other Non Commercial / Corporate Places (Non-Profit, Gov'T, Firehall)", 'Specialty - Other Non-Profit', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Religious
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Religious Facilities (Synagogue, Church, Convent, Mosque)', 'Specialty - Religious', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Specialty - Transportion
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Subway Station', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Go Station', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Train Tracks', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Train Admin Or Support Facility', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Go Train', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Passenger Train', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Bus Stop / Shelter / Loop', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Admin Or Support Facility', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Bus', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Regional Transit System Vehicle', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Subway Train', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Light Rail Transit Station', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Ttc Street Car', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Go Bus', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Passenger Train Station', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Location_Type - Unknown
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Unknown', 'Unknown', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Go Bus', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
dataframe_theft_bike['Location_Type'] = np.where(dataframe_theft_bike['Location_Type'] == 'Other Passenger Train Station', 'Specialty - Transportion', dataframe_theft_bike['Location_Type'])
###  Data Set - Column - Status - Transform - Binary Classification | Int
dataframe_theft_bike['Status'] = [1 if b=='RECOVERED' else 0 for b in dataframe_theft_bike.Status]
dataframe_theft_bike["Status"] = dataframe_theft_bike["Status"].astype(str).astype(int)

statistics()

### Data Set - Model - Features - Initial - Location | Bicycle Specification
list_columns_model = ['Location_Type', 'Premise_Type', 'Hood_ID', 'Bike_Type', 'Cost_of_Bike', 'Status']
dataframe_theft_bike_model = dataframe_theft_bike[list_columns_model]

### Data Set - Model - Column - Names | Values | Type | Info
print("\n\nData Set - Model - Columns")
print(dataframe_theft_bike_model.columns.values)
print("\n\nData Set - Model - Data Types")
print(dataframe_theft_bike_model.dtypes)
print("\n\nData Set - Model - Info")
print(dataframe_theft_bike_model.info())
print("\n\nData Set - Model - Description")
dataframe_theft_bike_model.describe()

### ----- Data Set - Missing Data Management ----- ###

### Data Set - Null Values
print(dataframe_theft_bike.isnull().sum())

print(dataframe_theft_bike['Bike_Model'].isnull().sum())
print(dataframe_theft_bike['Bike_Colour'].isnull().sum())
print(dataframe_theft_bike['Cost_of_Bike'].isnull().sum())

print(len(dataframe_theft_bike) - dataframe_theft_bike.count())

### Data Set - Missing Values - Drop
### Data Set - Missing Values - Drop - Status - Unknown
dataframe_theft_bike.drop(dataframe_theft_bike.index[dataframe_theft_bike['Status'] == 'UNKNOWN'], inplace = True)

### Data Set - Missing Values - Drop - Cost_of_Bike - Null
dataframe_theft_bike.loc[:,('Cost_of_Bike')].dropna(axis=0,how='any',inplace=True) 

### Data Set - Missing Values - Alternative - Fill Null - Cost_of_Bike - Mean
dataframe_theft_bike['Cost_of_Bike'].fillna(dataframe_theft_bike['Cost_of_Bike'].mean(),inplace=True)

### Data Set - Null Values
print(dataframe_theft_bike.isnull().sum())

### ----- Data Set - Categorical Management ----- ###

### List - Categorical Columns
list_categoricals = []
for col, col_type in dataframe_theft_bike_model.dtypes.iteritems():
    if col_type == 'O':
        list_categoricals.append(col)
print(list_categoricals)

### List - Transformation - Categorical Columns Values To Numeric Values - Using pd.get_dummies
dataframe_theft_bike_model_ohe = pd.get_dummies(dataframe_theft_bike_model, columns=list_categoricals, dummy_na=False)
print(dataframe_theft_bike_model_ohe.head())
print(dataframe_theft_bike_model_ohe.columns.values)
print(len(dataframe_theft_bike_model_ohe) - dataframe_theft_bike_model_ohe.count())

### Data Set - Column - Names | Values | Type | Info | Description/Summaries
print("\n\nData Set - Columns")
print(dataframe_theft_bike_model_ohe.columns.values)
print("\n\nData Set - Data Types")
print(dataframe_theft_bike_model_ohe.dtypes)
print("\n\nData Set - Info")
print(dataframe_theft_bike_model_ohe.info())
print("\n\nData Set - Description")
dataframe_theft_bike_model_ohe.describe()

### ----- Data Set - Standardization - i.e. Mean of Zero ----- ###

### Data Set - Column Names
names = dataframe_theft_bike_model_ohe.columns
### Scaler Object
scaler = preprocessing.StandardScaler()
### Scaler Object - Fit - Data Set
dataframe_scaled_theft_bike_model_ohe = scaler.fit_transform(dataframe_theft_bike_model_ohe)
dataframe_scaled_theft_bike_model_ohe = pd.DataFrame(dataframe_scaled_theft_bike_model_ohe, columns=names)

##################################################
# d. Managing imbalanced classes if needed.  Check here for info: https://elitedatascience.com/imbalanced-classes

### ----- Data Set - Imbalanced Classes Management ----- ###

### Go To: 3. Predictive Model Building - Model 02 - Random Forest Classifier
### Go To: 3. Predictive Model Building - Model 04 - SVC - Penalized-SVM

### Imbalanced Classes Management - Down-sample Majority Class
### Separate majority and minority classes
dataframe_majority = dataframe_scaled_theft_bike_model_ohe[dataframe_scaled_theft_bike_model_ohe['Status']<0]
dataframe_minority = dataframe_scaled_theft_bike_model_ohe[dataframe_scaled_theft_bike_model_ohe['Status']>1]
 
### Downsample majority class
dataframe_majority_downsampled = resample(dataframe_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(dataframe_minority),     # to match minority class
                                 random_state=0) # reproducible results

 
### Combine minority class with downsampled majority class
dataframe_scaled_theft_bike_model_ohe_downsampled = pd.concat([dataframe_majority_downsampled, dataframe_minority])
 
### Display new class counts
dataframe_scaled_theft_bike_model_ohe_downsampled.Status.value_counts()

##################################################
# b. Feature selection – use pandas and sci-kit learn.

### Data Set - X (inputs, predictor) | Y(output, predicted)
dataframe_theft_bike_model_ohe_columns=dataframe_scaled_theft_bike_model_ohe.columns.values.tolist()
y=['Status']
x=[i for i in dataframe_theft_bike_model_ohe_columns if i not in y ]
type(y)
type(x)

### ----- Data Set - Feature Selection ----- ###
##### Unknown label type: 'continuous'
##### https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
lab_encoder = preprocessing.LabelEncoder()
y_int = lab_encoder.fit_transform(dataframe_scaled_theft_bike_model_ohe[y].values.ravel())

model = LogisticRegression()
rfe = RFE(model, 12)
rfe = rfe.fit(dataframe_scaled_theft_bike_model_ohe[x],dataframe_scaled_theft_bike_model_ohe[y].values.ravel().astype('int'))
print(rfe.support_)
print(rfe.ranking_)

### Data Set - x (inputs, predictor) | y (output, predicted) - Update
list_features = []
counter=0
for i in rfe.support_:
    if i == True:
        list_features.append(x[counter])
    counter+=1  
    
print("\n\nData Set - Feature - Selection")
print(list_features)
x=dataframe_scaled_theft_bike_model_ohe_downsampled[list_features]
y=dataframe_scaled_theft_bike_model_ohe_downsampled['Status']
type(y)
type(x)
x = pd.DataFrame(x)
y = pd.DataFrame(y)
y = y.astype('int')

##################################################
# c. Train, Test data spliting – use numpy, sci-kit learn.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

"--------------------------------------------------"
# 3. Predictive Model Building
"--------------------------------------------------"

##################################################
# a. Use logistic regression and decision trees  as a minimum– use scikit learn

### ----- Model 01 - Logistic Regression ----- ###
##### DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
##### https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
### Model 01 - Train Model
model01 = linear_model.LogisticRegression(solver='lbfgs')
model01.fit(x_train, y_train.values.ravel())

### Model 01 - Test Data
list_model01_class_probabilities = model01.predict_proba(x_test.astype('int'))
list_model01_class_predictions = model01.predict(x_test.astype('int'))
print("Model 01 - Class Probabilities")
print(list_model01_class_probabilities)
print("Model 01 - Class Predictions")
print(list_model01_class_predictions)

### ----- Model 02 - Random Forest Classifier ----- ###
 
### Model 02 - Train Model
model02 = RandomForestClassifier()
model02.fit(x_train, y_train.values.ravel())

### Model 02 - Test Data
list_model02_class_probabilities = model02.predict_proba(x_test.astype('int'))
list_model02_class_predictions = model02.predict(x_test.astype('int'))
print("Model 02 - Class Probabilities")
print(list_model02_class_probabilities)
print("Model 02 - Class Predictions")
print(list_model02_class_predictions)

### ----- Model 03 - Decision Tree Classifier ----- ###
 
### Model 03 - Train Model
model03 = DecisionTreeClassifier()#(criterion="entropy", max_depth=23)
model03.fit(x_train, y_train)

### Model 03 - Test Data
list_model03_class_probabilities = model03.predict_proba(x_test.astype('int'))
list_model03_class_predictions = model03.predict(x_test.astype('int'))
print("Model 03 - Class Probabilities")
print(list_model03_class_probabilities)
print("Model 03 - Class Predictions")
print(list_model03_class_predictions)

### ----- Model 04 - SVC - Penalized-SVM ----- ###
# Train model
model04 = SVC(kernel='linear', 
            class_weight='balanced', # penalize
            probability=True)
model04.fit(x_train, y_train.values.ravel())

### Model 04 - Test Data
list_model04_class_probabilities = model04.predict_proba(x_test.astype('int'))
list_model04_class_predictions = model04.predict(x_test.astype('int'))
print("Model 04 - Class Probabilities")
print(list_model04_class_probabilities)
print("Model 04 - Class Predictions")
print(list_model04_class_predictions)

"--------------------------------------------------"
# 4. Model Scoring and Evaluation
"--------------------------------------------------"

##################################################
# a. Present results as scores, confusion matrices and ROC - use sci-kit learn

### ----- Model - Unique | Scores ----- ###

print("\n\nModel 01 - Unique | Scores")
print( np.unique( list_model01_class_predictions ) )
print("\n\nModel 02 - Unique | Scores")
print( np.unique( list_model02_class_predictions ) )
print("\n\nModel 03 - Unique | Scores")
print( np.unique( list_model03_class_predictions ) )
print("\n\nModel 04 - Unique | Scores")
print( np.unique( list_model04_class_predictions ) )

### ----- Model - Accuracy | Scores ----- ###

print("\n\nModel 01 - Accuracy | Scores")
print(metrics.accuracy_score(y_test, list_model01_class_predictions))
print("\n\nModel 02 - Accuracy | Scores")
print(metrics.accuracy_score(y_test, list_model02_class_predictions))
print("\n\nModel 03 - Accuracy | Scores")
print(metrics.accuracy_score(y_test, list_model03_class_predictions))
print("\n\nModel 04 - Accuracy | Scores")
print(metrics.accuracy_score(y_test, list_model04_class_predictions))

### Model - Cross Validation
model_scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), x, y.values.ravel(), scoring='accuracy', cv=10)
print("\n\nModel 01 - Accuracy | Scores | Cross Validation")
print(model_scores)
print("\n\nModel 01 - Accuracy | Scores | Mean")
print(model_scores.mean())

### ----- Model - Confusion Matrix ----- ###

##### TypeError: 'numpy.ndarray' object is not callable
##### https://stackoom.com/question/3mJsb/sklearn-confusion-matrix-TypeError-numpy-ndarray-对象不可调用
def confusion_Matrix(class_probabilities):
    from sklearn.metrics import confusion_matrix
    class_probabilities=class_probabilities[:,1]
    dataframe_model_recovery_probability=pd.DataFrame(class_probabilities)
    dataframe_model_recovery_probability['predict']=np.where(dataframe_model_recovery_probability[0]>=0.05,0,9)
    y_values = np.array(y_test['Status'])
    y_prediction = np.array(dataframe_model_recovery_probability['predict'])
    print("\n\nModel - Confusion Matrix")
    cfm = confusion_matrix(y_values, y_prediction)
    print(cfm)
    ### Model - Confusion Matrix - Heat Map
    ax = plt.subplot()
    sns.heatmap(cfm, annot=True, ax = ax); #annot=True to annotate cells
    return

##### Note: Run one by one
confusion_Matrix(list_model01_class_probabilities)
confusion_Matrix(list_model02_class_probabilities)
confusion_Matrix(list_model03_class_probabilities)
confusion_Matrix(list_model04_class_probabilities)

### ----- Model - Area Under ROC Curve (AUROC) ----- ###

list_model01_class_probabilities_positive = [p[1] for p in list_model01_class_probabilities]
list_model02_class_probabilities_positive = [p[1] for p in list_model02_class_probabilities]
list_model03_class_probabilities_positive = [p[1] for p in list_model03_class_probabilities]
list_model04_class_probabilities_positive = [p[1] for p in list_model04_class_probabilities]
##### ValueError: Found input variables with inconsistent numbers of samples: [21584, 6476]
##### https://datascience.stackexchange.com/questions/20199/train-test-split-error-found-input-variables-with-inconsistent-numbers-of-sam
##### https://www.edureka.co/blog/python-list-length/#:~:text=There%20is%20a%20built-in,length%20of%20the%20given%20list.
### Model 01 - Class Predictions
print("\n\nModel 01 - Area Under ROC Curve (AUROC)")
print( roc_auc_score(y_test, list_model01_class_probabilities_positive) )
### Model 02 - Class Predictions
print("\n\nModel 02 - Area Under ROC Curve (AUROC)")
print( roc_auc_score(y_test, list_model02_class_probabilities_positive) )
### Model 03 - Class Predictions
print("\n\nModel 03 - Area Under ROC Curve (AUROC)")
print( roc_auc_score(y_test, list_model03_class_probabilities_positive) )
### Model 04 - Class Predictions
print("\n\nModel 04 - Area Under ROC Curve (AUROC)")
print( roc_auc_score(y_test, list_model04_class_probabilities_positive) )

##################################################
# b. Select and recommend the best performing model 

### Model - Select - AUC Score - Max

list_auc_score = []
list_auc_score.append(roc_auc_score(y_test, list_model01_class_probabilities_positive))
list_auc_score.append(roc_auc_score(y_test, list_model02_class_probabilities_positive))
list_auc_score.append(roc_auc_score(y_test, list_model03_class_probabilities_positive))
list_auc_score.append(roc_auc_score(y_test, list_model04_class_probabilities_positive))

list_models = [model01, model02, model03, model04]
model = list_models[list_auc_score.index(max(list_auc_score))]

"--------------------------------------------------"
# 5. Deploying the model
"--------------------------------------------------"

##################################################
# a. Using flask framework arrange to turn your selected machine-learning model into an API.

### Go To: C:/COMP309-003 - Project 02/COMP309-003-003 - Project 02 - Flask.py

##################################################
# b. Using pickle module arrange for Serialization & Deserialization of your model.

### ----- Model - Serialization ----- ###

with open(path + '/model_prediction_recovery_bicycle.pkl', 'wb') as file:
    pickle.dump(model, file)
    print("Model - Serialized")
    
with open(path + '/model_prediction_recovery_bicycle_columns.pkl', 'wb') as file:
    pickle.dump(list_features, file)
    print("Model - Columns - Serialized")

##################################################
# c. Build a client to test your model API service. Use the test data, which was not previously used to train the module. You can use simple Jinja HTML templates with or without Java script, REACT or any other technology but at minimum use POSTMAN Client API.

### Go To: C:/COMP309-003 - Project 02/COMP309-003-003 - Project 02 - Flask.py