import pandas as pd
from plotnine import *


# Import Data
heart_data = pd.read_csv('hd_data.csv')
# print(heart_data.describe())

# Exploratory Analysis to assess variable relationships visually
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