### Output needs to be organized so it can be readily interpretable ### 
import pandas as pd
from scipy.stats import ttest_ind

# Import data
heart_data = pd.read_csv('hd_data.csv')
print(heart_data.describe())

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