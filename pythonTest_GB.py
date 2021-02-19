#Import future packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

#Data of the experiments cannot be shared due to confidentiality agreement.
#This is an example to import the experiments. (Parquet files were used).
#Each file was composed by the following columns, representing the features. ('V1','V2','V3','V4','V5','V6','I','V','PW','T1','T2')
#The rows represented the instances of the experimentes.


df=pd.concat([pd.read_parquet('allExperimentsT2.parquet', engine='pyarrow'),\
              pd.read_parquet('allExperimentsT6.parquet', engine='pyarrow'),\
              pd.read_parquet('allExperimentsT4.parquet', engine='pyarrow'),\
              pd.read_parquet('allExperimentsT5.parquet', engine='pyarrow')])
    
df.head()

#Selection of features and the target value. The target was calculated with Coulomb counting.
features=['V2','I','V','PW','T1','T2']
target=['SOC']

#The experiments were stablished with a string label for the type of experiment and another for the temperature used in the experiment (One column for each label)

experimentNames_df=df.experimentName.unique()
temperatureTest_df=df.temperatureTest.unique()


#Train test random splitting

from sklearn.model_selection import train_test_split
import random as rdm

rdm.seed(13)

x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0,shuffle = False)

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

#Definition of the base model

params = {'max_depth': None,
          'min_samples_split': 5,
          'learning_rate': 0.05,
          'loss': 'lad'}
gbrt = GradientBoostingRegressor(**params)

n_estimators = 190



#Iteration to find the best number of estimators

valVector=[]
min_val_error = float(" inf") 
error_going_up = 0 
for n_estimators in range( 1, 202,20): 
    gbrt.n_estimators = n_estimators 
    gbrt.fit( x_train, y_train) 
    y_pred = gbrt.predict(x_validation) 
    val_error = mean_squared_error( y_validation, y_pred) 
    valVector=[valVector,val_error]
    if val_error < min_val_error: 
        min_val_error = val_error 
        error_going_up =0
    else:
        error_going_up += 1 
        if error_going_up == 4: 
            break



#Calculate the error with the test-set

mae = mean_absolute_error(y_test, gbrt.predict(x_test))
mse = mean_squared_error(y_test, gbrt.predict(x_test))
rmse = sqrt(mean_squared_error(y_test, gbrt.predict(x_test)))



#Plot the test-set


x = x_test
y_true = y_test.to_numpy()
y_pred = gbrt.predict(x)

plt.figure(figsize=(15,5))
plt.plot(y_true, label='true')
plt.plot(y_pred, label='pred')
plt.ylabel('SOC')
plt.legend()
plt.show()


#Plot 6 random experiments taking into account all information

numberOfTest=len(experimentNames_df)*len(temperatureTest_df)

valTestExperimentsNumbers=[]

import random as rdm

rdm.seed(13)

plt.rcParams['font.size'] = '16'

for i in range(0,round(0.3*numberOfTest)):
    newTest=[rdm.randint(0,len(experimentNames_df)-1),
                         rdm.randint(0,len(temperatureTest_df)-1)]
    while newTest in valTestExperimentsNumbers:
        newTest=[rdm.randint(0,len(experimentNames_df)-1),
                         rdm.randint(0,len(temperatureTest_df)-1)]
    else:            
        valTestExperimentsNumbers.append(newTest)
        
for j in range(0,round(0.3*numberOfTest)):
    
    selected_df=df.loc[(df['experimentName'] == experimentNames_df[valTestExperimentsNumbers[6][0]])
                   & (df['temperatureTest'] == temperatureTest_df[valTestExperimentsNumbers[6][1]])]  
    
    x = selected_df[features].to_numpy()
    y_true = selected_df[target].to_numpy()
    y_pred = gbrt.predict(x)
    plt.figure(figsize=(15,5))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Prediction')
    plt.ylabel('SOC [p.u]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid()
    plt.show()
    
    #%%Voltage_DOD
    
    #np.flip(y_true,axis=1)
    
    x = selected_df[features].to_numpy()
    y_true = selected_df[target].to_numpy()
    Vt = selected_df['V'].to_numpy()
    y_pred = gbrt.predict(x)
    plt.figure(figsize=(15,5))
    plt.plot(y_true,Vt, label='True')
    plt.plot(y_pred,Vt, label='Prediction')
    plt.xlabel('DOD [p.u]')
    plt.ylabel('Voltage [V]')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid()
    plt.show()
    
    #%% Residuals
    
    res=((y_pred-selected_df['SOC'].to_numpy())/selected_df['SOC'].to_numpy())*100
    plt.figure(figsize=(15,5))
    plt.rcParams['font.size'] = '16'
    plt.plot(res)
    plt.ylabel('Estimate error [%]')
    plt.xlabel('Time [s]')
    plt.grid()
    plt.show()
    
    #%% Histogram
    
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(15,5))
    plt.hist(res, density=False, bins=50)
    plt.grid()
    plt.show()
    plt.ylabel('Frequency')
    plt.xlabel('Estimate error [%]')
