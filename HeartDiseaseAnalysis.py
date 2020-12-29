##### HEART DISEASE & LOGISTIC REGRESSION #####
# Imported Packages
import numpy as np
import pandas
import os
from plotnine import *
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.datasets.base import Bunch
import csv
from DataFormatting import convert_my_data


# ImportED heart disease dataset
heart_data = pandas.read_csv('hd_data.csv')
print(heart_data.describe())

### EXPLORATORY ANALYSIS AND IDENTIFICATION OF FEATURES OF INTEREST
# Create several plots to view rough relationship for mortality variable vs other features of dataset
plt1 = ggplot(heart_data, aes(x = 'ejection_fraction', y = 'creatinine_phosphokinase', color = "DEATH_EVENT", size = "smoking")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Ejection Fraction vs Creatinine Phosphokinase vs Smoking Status vs Death Event',
         x = "Ejection Fraction",
          y = "Creatinine Phosphokinase")
plt2 = ggplot(heart_data, aes(x = 'ejection_fraction', y = 'creatinine_phosphokinase', color = "DEATH_EVENT", size = "high_blood_pressure")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Ejection Fraction vs Creatinine Phosphokinase vs High Blood Pressure vs Death Event',
         x = "Ejection Fraction",
          y = "Creatinine Phosphokinase")
plt3 = ggplot(heart_data, aes(x = 'age', y = 'creatinine_phosphokinase', color = "DEATH_EVENT", size = "smoking")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Age vs Creatinine Phosphokinase vs Smoking Status vs Death Event',
         x = "Age",
         y = "Creatinine Phosphokinase")
plt4 = ggplot(heart_data, aes(x = 'age', y = 'creatinine_phosphokinase', color = "DEATH_EVENT", size = "high_blood_pressure")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Age vs Creatinine Phosphokinase vs High Blood Pressure vs Death Event',
         x = "Age",
         y = "Creatinine Phosphokinase")

# Using too many variables, focus only on 2 continuous variables and DEATH_EVENT
plt5 = ggplot(heart_data, aes(x = 'age', y = 'creatinine_phosphokinase', color = "DEATH_EVENT")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Age vs Creatinine Phosphokinase vs Death Event',
         x = "Age",
         y = "Creatinine Phosphokinase")

plt6 = ggplot(heart_data, aes(x = 'ejection_fraction', y = 'serum_creatinine', color = "DEATH_EVENT")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Ejection Fraction vs Serum Creatinine vs Death Event',
         x = "Ejection Fraction",
         y = "Serum Creatinine")

# Some pattern seen, add smoking as categorical predictor
plt7 = ggplot(heart_data, aes(x = 'ejection_fraction', y = 'serum_creatinine', color = "DEATH_EVENT", size = "smoking")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Ejection Fraction vs Serum Creatinine vs Smoking vs Death Event',
         x = "Ejection Fraction",
         y = "Serum Creatinine")
# No further discernment of pattern with addition of smoking or high blood pressure

# Try adding age to graph
plt8 = ggplot(heart_data, aes(x = 'ejection_fraction', y = 'serum_creatinine', color = "DEATH_EVENT", size = "age")) + geom_point() +\
    scale_color_gradient(low="red",high="blue") +\
        labs(title = 'Ejection Fraction vs Serum Creatinine vs Age vs Death Event',
         x = "Ejection Fraction",
         y = "Serum Creatinine")

# Save all plots to working directory
plotList = [plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8]
for i in range(len(plotList)):
    filename = "plt" + str(i+1) + ".png"
    plotList[i].save(filename)

# Compare mean ejection fraction & creatinine phosphokinase between death event groups
deathGrp = heart_data.loc[(heart_data['DEATH_EVENT'] > 0)]
aliveGrp = heart_data.loc[(heart_data['DEATH_EVENT'] == 0)]
numDead = len(deathGrp)
numAlive = len(aliveGrp)

# Calculate t-test for ejection fraction, serum creatinine, creatinine phosphokinase, & age between the alive and dead groups
ejectionTtest = ttest_ind(deathGrp['ejection_fraction'], aliveGrp['ejection_fraction'])
creatininePhTtest = ttest_ind(deathGrp['creatinine_phosphokinase'], aliveGrp['creatinine_phosphokinase'])
serumCreatinineTtest = ttest_ind(deathGrp['serum_creatinine'], aliveGrp['serum_creatinine'])
ageTtest = ttest_ind(deathGrp['age'], aliveGrp['age'])
serumSoTtest = ttest_ind(deathGrp['serum_sodium'], aliveGrp['serum_sodium'])
plateletsTtest = ttest_ind(deathGrp['platelets'], aliveGrp['platelets'])
timeTtest = ttest_ind(deathGrp['time'], aliveGrp['time'])
print("ejection ttest: " + str(ejectionTtest)) # Found to be significant
print("creatininePH ttest: " + str(creatininePhTtest)) # Not found to be significant
print("serumCre: "+str(serumCreatinineTtest)) # Found to be significant
print("Age Ttest: "+str(ageTtest)) # Found to be significant
print("serumsodium ttest: "+str(serumSoTtest)) # Found to be significant
print("Platelets ttest: "+str(plateletsTtest)) # not found to be significant
print("Time ttest: "+str(timeTtest)) # found to be significant
print("BONFERRONI")



### PRE-PROCESSING (ADDED POST COMPLETION OF SCRIPT TO NORMALIZE ALL FEATURES THAT ARE CONTINUOUS VARIABLES) ###
# Normalize continuous data using min-max normalization (this should have been completed more efficiently)
heart_data['age'] = (heart_data['age']-heart_data['age'].min())/(heart_data['age'].max()-heart_data['age'].min())
heart_data['creatinine_phosphokinase'] = (heart_data['creatinine_phosphokinase']-heart_data['creatinine_phosphokinase'].min())/(heart_data['creatinine_phosphokinase'].max()-heart_data['creatinine_phosphokinase'].min())
heart_data['ejection_fraction'] = (heart_data['ejection_fraction']-heart_data['ejection_fraction'].min())/(heart_data['ejection_fraction'].max()-heart_data['ejection_fraction'].min())
heart_data['serum_creatinine'] = (heart_data['serum_creatinine']-heart_data['serum_creatinine'].min())/(heart_data['serum_creatinine'].max()-heart_data['serum_creatinine'].min())
heart_data['serum_sodium'] = (heart_data['serum_sodium']-heart_data['serum_sodium'].min())/(heart_data['serum_sodium'].max()-heart_data['serum_sodium'].min())
heart_data['platelets'] = (heart_data['platelets']-heart_data['platelets'].min())/(heart_data['platelets'].max()-heart_data['platelets'].min())
heart_data['time'] = (heart_data['time']-heart_data['time'].min())/(heart_data['time'].max()-heart_data['time'].min())

### FORMATTING OF DATASET FOR USE W/ SKLEARN FEATURES & APPLICATION OF LOGISTIC REGRESSION ###
### MODEL 1 ### 
# Format dataframe to features of interest and outcome to predict, use first significant vars
data_to_model = heart_data[['ejection_fraction','serum_creatinine', 'DEATH_EVENT']]
# data_to_model = data_to_model.drop(data_to_model.columns[0], axis=1) # Drop the excess index column
data_to_model.to_csv('convertData.csv')

# Convert the data to the correct array formatting
processedData = convert_my_data('convertData.csv')

# Set the X & y variable vectors to be input into LogisticRegression
X, y = processedData.data, processedData.target
trainingX, testingX, trainingY, testingY = train_test_split(X, y, test_size=0.25, random_state=10)

# Set regression variable and call LogisticRegression class
model_output = LogisticRegression(learningrate=0.001, nIter=1000)
model_output.gradientDescent(trainingX,trainingY)
model_prediction = model_output.predict(testingX)

# Identify how accurate my model is
modelAccuracy = np.sum(testingY==model_prediction) / len(testingY)
print("Logistic Regression model accuracy is: ", str(modelAccuracy)) # Logistic Regression model accuracy is:  0.88

### MODEL 2 (INCLUDES AGE) ###
# Format second dataframe that includes binary variables (to be based off previous plots)
data_to_model2 = heart_data[['age','ejection_fraction','serum_creatinine','DEATH_EVENT']]
# data_to_model2 = data_to_model2.drop(data_to_model2.columns[0], axis=1)
data_to_model2.to_csv('convertData2.csv')

# Convert the data to the correct array formatting
processedData2 = convert_my_data('convertData2.csv')

# Apply Logistic Regression alogorithm
X2, y2 = processedData2.data, processedData2.target
trainingX2, testingX2, trainingY2, testingY2 = train_test_split(X2, y2, test_size = 0.25, random_state = 10)

# set regression variable & call LogisticRegression class
model2_output = LogisticRegression(learningrate=0.001, nIter=1000)
model2_output.gradientDescent(trainingX2, trainingY2)
model2_prediction = model2_output.predict(testingX2)

# Calculate difference to determine accuracy of the second model
model2Accuracy = np.sum(testingY2==model2_prediction) / len(testingY2)
print("Logistic Regression model 2 accuracy is: ", str(model2Accuracy)) # Logistic Regression model 2 accuracy is:  0.8133333333333334

### MODEL 3 ( INCLUDES ALL CONTINUOUS VARIABLES) ###
data_to_model_final = heart_data[['age','ejection_fraction','serum_creatinine', 'serum_sodium','time','DEATH_EVENT']]
# data_to_model_final = data_to_model_final.drop(heart_data.columns[0], axis = 1)
data_to_model_final.to_csv('finalConvertData.csv')

# Convert to correct array formatting
final_processed_Data = convert_my_data('finalConvertData.csv')

# Apply Logistic regression
Xf, yf = final_processed_Data.data, final_processed_Data.target
trainingXf, testingXf, trainingYf, testingYf = train_test_split(Xf, yf, test_size = 0.25, random_state = 10)

# set regression variable & call LogisticRegression class
final_model_output = LogisticRegression(learningrate=0.001, nIter=1000)
final_model_output.gradientDescent(trainingXf, trainingYf)
final_model_prediction = final_model_output.predict(testingXf)

# Calculate difference to determine accuracy of the final model
final_model_Accuracy = np.sum(testingYf==final_model_prediction) / len(testingYf)
print("Logistic Regression final model accuracy is: ", str(final_model_Accuracy)) # Logistic Regression final model accuracy is:  0.8533333333333334

# As final model showed increase in accuracy, must check platelets/time/serum_sodium
# Found serum_sodium and time to be statistically significant, try model again and drop non-sig vars

### MODEL Z ### 
# Use only statistically significant cont.vars
data_to_modelZ = heart_data[['time', 'ejection_fraction','serum_creatinine','serum_sodium', 'DEATH_EVENT']]
# data_to_modelZ = data_to_modelZ.drop(data_to_model.columns[0], axis=1) # Drop the excess index column
data_to_modelZ.to_csv('convertDataZ.csv')

# Convert the data to the correct array formatting
processedDataZ = convert_my_data('convertDataZ.csv')

# Set the X & y variable vectors to be input into LogisticRegression
Xz, yz = processedDataZ.data, processedDataZ.target
trainingXZ, testingXZ, trainingYZ, testingYZ = train_test_split(Xz, yz, test_size=0.25, random_state=10)

# Set regression variable and call LogisticRegression class
modelZ_output = LogisticRegression(learningrate=0.001, nIter=1000)
modelZ_output.gradientDescent(trainingXZ,trainingYZ)
modelZ_prediction = modelZ_output.predict(testingXZ)

# Identify how accurate my model is
modelZAccuracy = np.sum(testingYZ==modelZ_prediction) / len(testingYZ)
print("Logistic Regression model accuracy is: ", str(modelZAccuracy)) # Logistic Regression model accuracy is:  0.76




