import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import linregress, logistic, kstest
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error


# User input
def get_integer_input(prompt: str, range_min = 0, range_max = sys.maxsize - 1):
    while True:
        while not (age := input(f"{prompt}:")).isdigit():
            print("Please provide numeric value \n")
        age = int(age)
        if age < range_min or age > range_max:
            print(f"Please provide value between: {range_min} and {range_max}")
        else:
            return age


def get_float_input(prompt: str, range_min = 0, range_max = sys.maxsize - 1):
    while True:
        try:
            feature = float(input(f"{prompt} "))
            if feature < range_min or feature > range_max:
                print(f"Please provide value between: {range_min} and {range_max}")
            else:
                return feature
        except ValueError:
            print("Please provide numeric value")


def get_generic_input(prompt: str, options: tuple):
    while (user_input := input(f"{prompt} ({'/'.join(options)}):").lower()) not in [option.lower() for option in options]:
        print("Please provide a valid option")
    first_letter = user_input[0].upper()
    user_input = user_input[1:]
    user_input = first_letter + user_input
    return user_input


def profile():
    gender = get_generic_input('What is your biological gender? ', ('male', 'female'))
    gender = gender_mapping[gender]                                                                # Gender mapping to numeric value: male = 0 and female = 1
    age = get_integer_input('What is your age?', range_max=120)                                                                                          
    valid_work_types = input_employment_mapping.copy()                                             # User can't be a child if they are older than 17, so we are removing this option from the list of work types
    if age > 17:
        valid_work_types.pop('Children')
    work_type = get_generic_input("What is your work type?", valid_work_types)
    input_employment_mapping[work_type] = 1
    work_type = input_employment_mapping.values()
    hypertension = get_generic_input("Do you have hypertension?", ("yes", "no"))                   # Hypertension mapping to numeric value: yes = 1, no = 0
    hypertension = input_hypertension_mapping[hypertension]   
    heart_disease = get_generic_input("Do you have any heart disease?", ("yes", "no"))
    heart_disease = input_heart_disease_mapping[heart_disease]                                     # Heart disease mapping to numeric value: yes = 1, no = 0
    ever_married = get_generic_input("Have you ever been married?", ("yes", "no"))                 # Marital status mapping to numeric value: yes = 1, no = 0
    ever_married = input_ever_married_mapping[ever_married]
    residence_type = get_generic_input("What is your residence type?", ("urban", "rural"))         # Residence type mapping to numeric value: urban = 1, rural = 0
    residence_type = input_residence_mapping[residence_type]
    avg_glucose_level = get_float_input("Enter your average glucose level:", range_min = 0, range_max = 2500)
    bmi = get_float_input("Enter your BMI:", range_min = 0, range_max = 200)
    smoking_status = get_generic_input("What is your smoking status?", ("never smoked", "formerly smoked", "smokes"))
    input_smoking_mapping[smoking_status] = 1
    smoking_status = input_smoking_mapping.values()                                                  

    scaled_age = (age - scaler.mean_[0]) / np.sqrt(scaler.var_[0])                                 #Scaling all non-binary features to match the format of the training data
    scaled_avg_glucose_level = (avg_glucose_level - scaler.mean_[1]) / np.sqrt(scaler.var_[1])    
    scaled_bmi = (bmi - scaler.mean_[2]) / np.sqrt(scaler.var_[2])          

    personal_data = [gender, scaled_age, hypertension, heart_disease, ever_married, residence_type, scaled_avg_glucose_level, scaled_bmi, *work_type, *smoking_status]

    return personal_data, age, avg_glucose_level, bmi


#We are using the Monte Carlo simulation to predict at which age the probability of risk will be higher
def MonteCarlo():
    slope_bmi, intercept_bmi, _, _, std_err_bmi = linregress(dataset['age'], dataset['bmi'])                            # Change of BMI depending on age
    slope_gluco, intercept_gluco, _, _, std_err_gluco = linregress(dataset['bmi'], dataset['avg_glucose_level'])        # Change of glucose level depending on BMI
    future_risks = {} #Dictionary age:risk of stroke in that age

    for age_step in range(start_age + 1, 130):

        yearly_probs = []

        for j in range(1000): 

            # Simulating data for one virtual patient
            sim_bmi = intercept_bmi + (slope_bmi * age_step) + np.random.normal(0, std_err_bmi)
            sim_glucose = intercept_gluco + (slope_gluco * sim_bmi) + np.random.normal(0, std_err_gluco)
            
            # Ensuring simulated values are within realistic bounds. Minimum BMI = 10 and minimum glucose level = 50 
            sim_bmi = max(10, sim_bmi)
            sim_glucose = max(50, sim_glucose)

            # Scaling data to match the format of the training data
            scaled_vals = scaler.transform([[age_step, sim_glucose, sim_bmi]])
            s_age, s_gluco, s_bmi = scaled_vals[0]

            # Building random vector for a virtual patient
            sim_profile = [
                PROFILE[0],      #gender
                s_age,           #scaled age
                PROFILE[2],      #hypertension
                PROFILE[3],      #heart_disease
                PROFILE[4],      #ever_married
                PROFILE[5],      #residence_type
                s_gluco,         #scaled glucose
                s_bmi,           #scaled bmi
                *PROFILE[8:13],  #work_type 
                *PROFILE[13:16]  #smoking_status
            ]

            # Calculating 
            prob = model.predict_proba([sim_profile])[0][1]
            yearly_probs.append(prob)
        
        # The final future prognosis is a mean of yearly_probs
        future_risks[age_step] = np.mean(yearly_probs)

    return future_risks


def Logistic_distribution(age,s,mu):
    return 1/(1+np.exp(-(age-mu)/s))


def Log_probability_density_function(age):
    return (np.exp(-(age - mu) / (s))) / (s * (1 + np.exp(- (age-mu) / (s) ))**2)


def hazard_function(age):
    return 1 / (s * ( 1 + np.exp(-(age-mu)/s) ))


# Data preprocessing
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv').drop(columns=['id']).dropna()         # Loading dataset without id column and missing values
dataset = dataset[dataset['gender'] != 'Other']
dataset = dataset[dataset['smoking_status'] != 'Unknown']

gender_mapping = {'Male': 0, 'Female': 1}                                                         # Mapping necessary values
married_mapping = {'No': 0, 'Yes': 1}
residence_mapping = {'Rural': 0, 'Urban': 1}
input_employment_mapping = {'Private': 0, 'Self-employed': 0, 'Children': 0, 'Government job': 0, 'Never worked': 0}
input_smoking_mapping = {'Never smoked': 0, 'Formerly smoked': 0, 'Smokes': 0} 
input_hypertension_mapping = {'No': 0, 'Yes': 1}
input_heart_disease_mapping = {'No': 0, 'Yes': 1}
input_ever_married_mapping = {'No': 0, 'Yes': 1}
input_residence_mapping = {'Rural': 0, 'Urban': 1}

dataset['gender'] = dataset['gender'].map(gender_mapping)
dataset['ever_married'] = dataset['ever_married'].map(married_mapping)
dataset['Residence_type'] = dataset['Residence_type'].map(residence_mapping)

dataset = pd.get_dummies(dataset, columns=['work_type'], dtype=int)
dataset = pd.get_dummies(dataset, columns=['smoking_status'], dtype=int)

# Dataset divisions
data = dataset.drop(columns=['stroke'])
result = dataset['stroke']
training_set, testing_set, training_result, testing_result = train_test_split(data, result, test_size=0.2, random_state=42)
scaler = StandardScaler()
cols_to_scale = ['age', 'avg_glucose_level', 'bmi']
training_set[cols_to_scale] = scaler.fit_transform(training_set[cols_to_scale])
testing_set[cols_to_scale] = scaler.transform(testing_set[cols_to_scale])

# Machine learning model (Logistic Regression)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(training_set, training_result)

# Evaluation
predictions = model.predict(testing_set)

# User's profile
PROFILE, start_age, start_glucose, start_bmi = profile()                     # Random vector with user input data, which will be used for prediction
current_disctribution = model.predict_proba([PROFILE])[0][1]                 # Probability that user is already in the group of patients with stroke

# All values of the empirical distribution for each age
past_risk = MonteCarlo()
DISTRIBUTION = {start_age:current_disctribution, **past_risk}

#We are running curve fit test for logistic distribution to find the probability density function
x = np.array(list(DISTRIBUTION.keys()))
y = np.array(list(DISTRIBUTION.values()))
parameters, _ = curve_fit(Logistic_distribution, x, y, bounds=([1e-5, 0], [100, 150]))        #Bounds are necessary - othervise hazard could be infinite
s, mu = parameters
prediction = Logistic_distribution(x, s, mu)
r2 = r2_score(y, prediction)
mse = mean_squared_error(y, prediction)
rmse = np.sqrt(mse)
print("Logistic distribution was assigned to your data. Here are the exact values of its parameters and features:\n")
print("s =", s)
print("mu =", mu)
print("R² =", r2)
print("MSE =", mse)
print("RMSE =", rmse)

#That proves that our random variable X (that says about the probability of stroke in exact time) has approximately logistic distribution
#Now we are calculating hazard
print("The current hazard of stroke for your age (", start_age, ") equals to:", hazard_function(start_age)*100, "%\n")
plt.plot(x, hazard_function(x)*100, label='Hazard Function')
plt.xlabel('Age')
plt.ylabel('Hazard of Stroke (%)')
plt.title('Hazard Function of Stroke by Age')
plt.legend()
plt.grid()
plt.show()