# HeartDiseaseLogisticRegression

View final report for conclusion and implementation of listed code. 

Dataset obtained from BMC Medical Informatics and Decision Making: 
"Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone"


# Supporting Scripts
> DataFormatting.py: Formats pandas dataframe to select for specific features
> ExploratoryStats.py: Demonstrates statistical relationships between key clinical metrics
> variablePlots.py: Visual demonstration of feature relationships
> LogisticRegression.py: Class defining script for the Logistic Regression model
> HeartDiseaseAnalysis.py: Implementation of the Logisitic Regression on 4 separate subsets of data

# DockerFile
> Docker container can be found here: docker pull thorpe7/hd_log_reg
> Latest is highly recommended
> For output retention, copy the output folder to local host directory: docker cp CONTAINER_NAME_OR_ID:/usr/src/app/Output $("pwd") 

# Next Steps
> This code simply demonstrates effective implementation of a classification model
> At this point in time, importing data through bind mount is discouraged as container still needs to be streamlined before new classification can be effectively utilized. 
> Persistance of trained model will also be incorporated at later date. 