##### HEART DISEASE & LOGISTIC REGRESSION #####
# Imported Packages
import numpy as np
import pandas
import os
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
import csv
from DataFormatting import convert_my_data


# ImportED heart disease dataset
heart_data = pandas.read_csv('hd_data.csv')

# Normalize continuous data using min-max normalization (Future update, please simplify implementation)
heart_data['age'] = (heart_data['age']-heart_data['age'].min())/(heart_data['age'].max()-heart_data['age'].min())
heart_data['creatinine_phosphokinase'] = (heart_data['creatinine_phosphokinase']-heart_data['creatinine_phosphokinase'].min())/(heart_data['creatinine_phosphokinase'].max()-heart_data['creatinine_phosphokinase'].min())
heart_data['ejection_fraction'] = (heart_data['ejection_fraction']-heart_data['ejection_fraction'].min())/(heart_data['ejection_fraction'].max()-heart_data['ejection_fraction'].min())
heart_data['serum_creatinine'] = (heart_data['serum_creatinine']-heart_data['serum_creatinine'].min())/(heart_data['serum_creatinine'].max()-heart_data['serum_creatinine'].min())
heart_data['serum_sodium'] = (heart_data['serum_sodium']-heart_data['serum_sodium'].min())/(heart_data['serum_sodium'].max()-heart_data['serum_sodium'].min())
heart_data['platelets'] = (heart_data['platelets']-heart_data['platelets'].min())/(heart_data['platelets'].max()-heart_data['platelets'].min())
heart_data['time'] = (heart_data['time']-heart_data['time'].min())/(heart_data['time'].max()-heart_data['time'].min())


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


file = open("modelPredictions.txt", 'w')
file.write("Logistic Regression model 1 accuracy is: " + str(modelAccuracy) + '\n')
file.write("Logistic Regression model 2 accuracy is: " + str(model2Accuracy) + '\n')
file.write("Logistic Regression final model accuracy is: " + str(final_model_Accuracy) + '\n')
file.write("Logistic Regression model accuracy is: " + str(modelZAccuracy))
file.close()

#  model_prediction, testingY2, model2_prediction, testingYf, final_model_prediction, testingYZ, modelZ_prediction]
df = pandas.DataFrame({
    "True M1" : testingY,
    "Pred M1" : model_prediction,
    "True M2": testingY2, 
    "Pred M2": model2_prediction,
    "True Mf": testingYf, 
    "Pred Mf": final_model_prediction, 
    "True Mz": testingYZ, 
    "Pred Mz": modelZ_prediction})

df.to_csv("trueVpred.csv", index=False)





