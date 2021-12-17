# Affective_Computing_Project_1
Python script that identifies (classify) pain from physiological data using Machine Learning. 

Your script needs to take a command line parameter to determine which data type will be run - python Project1.py <data_type>, where data_type is one of the following: 
* dia – Diastolic BP
* sys – Systolic BP
* eda – EDA
* res – Respiration
* all – Fusion of all data types

The script needs to read the CSV file with the colums: Subject ID, Data Type, Class, Data. Data is variable length.

The script will create the following hand-crafted features from the raw data:
* Mean
* Variance
* Entropy
* Min
* Max

These features will be used to build and train random forest to clasify pain vs no pain.
The script will permorf 10-fold cross-validation, and the output of the script is the confusion matrix, classification accuracy, precision and recall.

A PDF file with the results of the project is included.
