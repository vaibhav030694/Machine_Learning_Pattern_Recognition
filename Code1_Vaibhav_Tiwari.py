
#---------------------------------------------START - Download and import libraries or extensions.--------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from math import sqrt

#!pip install autoviz
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class  # step 02
AV = AutoViz_Class() #instantiaze the AV         # step 03
#!pip install dataprep
from dataprep.datasets import load_dataset
from dataprep.eda import create_report

#----------------------------------------------END - Download and import libraries or extensions-------------------------------------------------------------

#---------------------------------------------START - Read dataset.--------------------------------------------------------------
read_file_name = "fifa_dataset_raw.csv"
write_file_name = "fifa_dataset_cleaned.csv"

df = pd.read_csv(read_file_name,low_memory=False)
print(df.head())

#---------------------------------------------END - Read dataset.--------------------------------------------------------------

#---------------------------------------------START - Data Cleaning & Preparation.--------------------------------------------------------------

# Format column names
formatted_columns_name_arr = []
for column in df.columns:
    formatted_column = column.replace(" ", "_")
    formatted_columns_name_arr.append(formatted_column)

df.columns = formatted_columns_name_arr
print(df.columns)

# Filtered data based on players preferred positions and selected strikers and dropped others

df = df[df['Preferred_Positions'].str.contains("ST")]
print(df.head())


# Data formatting - Removed symbols(+,-,/) from data to have a numeric figures

striker_data_columns_arr = ['Age', 'Potential', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball_control',
                                'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Sprint_speed', 'Stamina',
                                'Strength', 'Vision']

for column in striker_data_columns_arr :
    entire_column_dataload_arr = []
    for row_data in df[column] :

        if str(row_data).__contains__("+") :
            format_row_data = str(row_data).split("+")
            format_row_data = int(format_row_data[0]) + int(format_row_data[1])
            entire_column_dataload_arr.append(format_row_data)
        elif str(row_data).__contains__("-") :
            format_row_data = str(row_data).split("-")
            format_row_data = int(format_row_data[0]) - int(format_row_data[1])
            entire_column_dataload_arr.append(format_row_data)
        elif str(row_data).__contains__("/"):
            # format_row_data = str(row_data).split("/")
            # format_row_data = int(format_row_data[0])
            entire_column_dataload_arr.append(0)
        else :
            entire_column_dataload_arr.append(row_data)
    df[column] = entire_column_dataload_arr

df.head

# Formatted columns to numeric columns from dataset

formatted_wage_arr = []
for wage in df.Wage:
  string_wage = str(wage)
  wage_formatting = string_wage.replace("€", "")
  wage_formatting = wage_formatting.replace("M", "000000")
  wage_formatting = wage_formatting.replace("K", "000")
  formatted_wage_arr.append(float(wage_formatting))

df["Wage"] = formatted_wage_arr
print(df["Wage"])

formatted_potential_arr = []
for value in df.Potential:
    formatted_potential_arr.append(float(value))
df["Potential"] = formatted_potential_arr
print(df["Potential"])


# Dropping columns those are not contributing in the analysis

columns_to_remove_from_raw_dataset = ['Unnamed:_0', 'Name', 'Photo', 'Nationality', 'Flag', 'Overall',
       'Club', 'Club_Logo', 'Special', 'Free_kick_accuracy', 'GK_diving', 'GK_handling', 'GK_kicking',
       'GK_positioning', 'GK_reflexes', 'Heading_accuracy', 'Interceptions',
       'Jumping', 'Long_passing', 'Long_shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short_passing', 'Shot_power',
       'Sliding_tackle', 'Standing_tackle',
       'Volleys', 'CAM', 'CB', 'CDM', 'CF', 'CM', 'ID',
       'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB',
       'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM',
       'RS', 'RW', 'RWB', 'ST']

df.drop(columns_to_remove_from_raw_dataset, axis=1, inplace=True)
#print()
print(df.columns)

#---------------------------------------------END - Data Cleaning & Preparation.--------------------------------------------------------------

#---------------------------------------------START - Write cleaned dataset to a file.--------------------------------------------------------------
df.to_csv(write_file_name)

#---------------------------------------------END - Write cleaned dataset to a file.--------------------------------------------------------------

#---------------------------------------------START - Read cleaned dataset again.--------------------------------------------------------------
df = pd.read_csv(write_file_name, low_memory=False)
print(df.head)

#---------------------------------------------END - Read cleaned dataset again.--------------------------------------------------------------

#---------------------------------------------START - Validations on dataset.--------------------------------------------------------------

#Analyzing column data type and null values
print(df.info())

#Checking stats for attributes
print(df.describe().T)

#dropping duplicate records
df.drop_duplicates()

#---------------------------------------------END - Validations on dataset.--------------------------------------------------------------


#---------------------------------------------START - Feature selection.--------------------------------------------------------------
model = df
# Selecting features for model training
model_features = model.loc[:, ['Age', 'Potential', 'Value', 'Wage', 'Acceleration', 'Aggression',
                                  'Agility', 'Balance', 'Ball_control',
                                  'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Sprint_speed', 'Stamina',
                                  'Strength', 'Vision','Preferred_Positions']]

# taking log value for value and storing as market value columns.
model_features['Player_potential'] = model['Potential']
df['Player_potential'] = np.log(model['Potential'])
#print(model_features)

# assigning independent features as X and dependent features as y and splitting the dataset in the ratio of 70:30
ind_X = model_features.loc[:,
        ['Age', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball_control','Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Sprint_speed',
          'Stamina','Strength', 'Vision']]

dep_y = model_features.loc[:, 'Player_potential']

print(ind_X)
print(dep_y)

#---------------------------------------------END - Feature selection.--------------------------------------------------------------

#---------------------------------------------START - Generate plots, graphs & report.--------------------------------------------------------------

#%matplotlib inline
sep = ","
dft = AV.AutoViz(
    write_file_name,
    sep=sep,
    depVar="",
    dfte=None,
    header=0,
    verbose=2,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=2000,
    max_cols_analyzed=20,
)
# Create report
create_report(df)

#---------------------------------------------END - Generate plots, graphs & report.--------------------------------------------------------------

##  Performing Multilinear Regression on dataset

#---------------------------------------------START - Performing Multilinear Regression on dataset.--------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(ind_X, dep_y, test_size=0.30, random_state=7)

# Basemodel
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

r2_score_test_data = metrics.r2_score(y_test, y_pred)
print("R2 Score of data ", r2_score_test_data)

mae_test = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error of data ", mae_test)

mape_test = metrics.mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute percentage Error of data ", mape_test)

msqe_test = metrics.mean_squared_error(y_test, y_pred)
print("Mean Square Error of data ", msqe_test)

msqle_test = metrics.mean_squared_log_error(y_test, y_pred)
print("Mean Square Log Error of data ", msqle_test)

rmse = sqrt(msqe_test)
print("Root Mean Square Error of data ", rmse)

#---------------------------------------------END - Performing Multilinear Regression on dataset.--------------------------------------------------------------

#---------------------------------------------START - Applying Gradient Descent optimization algorithm.--------------------------------------------------------------

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes values
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
         Parameters:
          X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def Multivariable_Linear_Regression(X,y,learningrate, iterations):
    """ Find the multivarite regression model for the data set
         Parameters:
          X:  independent variables matrix
          y: dependent variables matrix
          learningrate: learningrate of Gradient Descent
          iterations: the number of iterations
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y, (len(y), 1))
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        cost_value = 1/(2*len(y))*((y_pred - y)**2) #Calculate the loss for each training instance
        total = 0
        for i in range(len(y)):
            total += cost_value[i][0] #Calculate the cost function for each iteration
            #print(cost_value[i][0])
        cost_lst.append(total)
        #print(total)
    print("Iterations : " , iterations, " cost : " , cost_lst[1:])
    plt.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    plt.title('Cost function Graph')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    #plt.show()
    return theta, cost_lst


data_arr = []
target_arr = []
for index, row in df.iterrows():
    data_row_arr = []
    age_value = row['Age']
    potential_value = row['Player_potential']
    accelation_value = row['Acceleration']
    aggression_value = row['Aggression']
    agility_value = row['Agility']
    balance_value = row['Balance']
    ball_controll_value = row['Ball_control']
    composure_value = row['Composure']
    crossing_value = row['Crossing']
    curve_value = row['Curve']
    dribbling_value = row['Dribbling']
    finishing_value = row['Finishing']
    sprint_speed_value = row['Sprint_speed']
    stamina_value = row['Stamina']
    strength_value = row['Strength']
    vision_value = row['Vision']

    data_row_arr.append(int(age_value))
    data_row_arr.append(int(accelation_value))
    data_row_arr.append(int(aggression_value))
    data_row_arr.append(int(agility_value))
    data_row_arr.append(int(balance_value))
    data_row_arr.append(int(ball_controll_value))
    data_row_arr.append(int(composure_value))
    data_row_arr.append(int(crossing_value))
    data_row_arr.append(int(curve_value))
    data_row_arr.append(int(dribbling_value))
    data_row_arr.append(int(finishing_value))
    data_row_arr.append(int(sprint_speed_value))
    data_row_arr.append(int(stamina_value))
    data_row_arr.append(int(strength_value))
    data_row_arr.append(int(vision_value))

    data_arr.append(data_row_arr)

    target_arr.append(potential_value)

sc = StandardScaler()
data_transform = sc.fit_transform(data_arr)

cost_value,cost_function_mse_arr = Multivariable_Linear_Regression(data_transform, target_arr, 0.005, 30000)

index = 0
cost_arr = []
for val in cost_value:
    cost_arr.append(float(val))

minimum_cost_value_mse = min(i for i in cost_function_mse_arr if i > 0)

print("Position:", cost_function_mse_arr.index(minimum_cost_value_mse))
print("Value:", minimum_cost_value_mse)
print("Cost by Gradient Descent Algorithm : ", cost_arr)

#---------------------------------------------START - Applying Gradient Descent optimization algorithm.--------------------------------------------------------------
