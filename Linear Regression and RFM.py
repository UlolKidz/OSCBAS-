import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

file_path = r"C:\Users\depri\Downloads\study_performance .csv"
data = pd.read_csv(file_path)

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop(columns=['math_score', 'reading_score', 'writing_score'])
y_math = data_encoded['math_score']
y_reading = data_encoded['reading_score']
y_writing = data_encoded['writing_score']

X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(X, y_math, test_size=0.2, random_state=42)
X_train_reading, X_test_reading, y_train_reading, y_test_reading = train_test_split(X, y_reading, test_size=0.2, random_state=42)
X_train_writing, X_test_writing, y_train_writing, y_test_writing = train_test_split(X, y_writing, test_size=0.2, random_state=42)

math_model = LinearRegression().fit(X_train_math, y_train_math)
reading_model = LinearRegression().fit(X_train_reading, y_train_reading)
writing_model = LinearRegression().fit(X_train_writing, y_train_writing)

math_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': math_model.coef_})
reading_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': reading_model.coef_})
writing_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': writing_model.coef_})

math_predictions = math_model.predict(X_test_math)
reading_predictions = reading_model.predict(X_test_reading)
writing_predictions = writing_model.predict(X_test_writing)

math_rmse = mean_squared_error(y_test_math, math_predictions, squared=False)
reading_rmse = mean_squared_error(y_test_reading, reading_predictions, squared=False)
writing_rmse = mean_squared_error(y_test_writing, writing_predictions, squared=False)

print("Math Coefficients:\n", math_coefficients)
print("\nReading Coefficients:\n", reading_coefficients)
print("\nWriting Coefficients:\n", writing_coefficients)

print("\nModel Performance:")
print(f"Math RMSE: {math_rmse:.2f}")
print(f"Reading RMSE: {reading_rmse:.2f}")
print(f"Writing RMSE: {writing_rmse:.2f}")

math_coefficients['Score Type'] = 'Math'
reading_coefficients['Score Type'] = 'Reading'
writing_coefficients['Score Type'] = 'Writing'

all_coefficients = pd.concat([math_coefficients, reading_coefficients, writing_coefficients])

plt.figure(figsize=(14, 8))
sns.barplot(x='Coefficient', y='Feature', hue='Score Type', data=all_coefficients, palette='viridis')
plt.title("Impact of Features on Test Scores")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.legend(title="Score Type")
plt.tight_layout()
plt.show()

rmse_values = pd.DataFrame({'Score Type': ['Math', 'Reading', 'Writing'], 'RMSE': [math_rmse, reading_rmse, writing_rmse]})

plt.figure(figsize=(8, 5))
sns.barplot(x='Score Type', y='RMSE', data=rmse_values, palette='plasma')
plt.title("Model Performance (RMSE)")
plt.xlabel("Score Type")
plt.ylabel("Root Mean Square Error")
plt.tight_layout()
plt.show()

corr_matrix = data_encoded.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Features and Scores")
plt.tight_layout()
plt.show()

rf_math = RandomForestRegressor(random_state=42).fit(X_train_math, y_train_math)
rf_reading = RandomForestRegressor(random_state=42).fit(X_train_reading, y_train_reading)
rf_writing = RandomForestRegressor(random_state=42).fit(X_train_writing, y_train_writing)

math_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_math.feature_importances_}).sort_values(by='Importance', ascending=False)
reading_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_reading.feature_importances_}).sort_values(by='Importance', ascending=False)
writing_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_writing.feature_importances_}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=math_importances, palette='Blues_d')
plt.title("Math Score Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=reading_importances, palette='Blues_d')
plt.title("Reading Score Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=writing_importances, palette='Blues_d')
plt.title("Writing Score Feature Importance")
plt.tight_layout()
plt.show()

def prepare_input_data(gender, ethnicity, parent_education, test_prep, lunch, training_columns):
    input_data = {
        'gender': [gender],
        'ethnicity': [ethnicity],
        'parent_education': [parent_education],
        'test_preparation_course': [test_prep],
        'lunch': [lunch]
    }

    input_df = pd.DataFrame(input_data)
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_df_encoded = input_df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return input_df_encoded

gender = 'female'
ethnicity = 'group E'
parent_education = 'some college'
test_prep = 'completed'
lunch = 'standard'

input_data = prepare_input_data(gender, ethnicity, parent_education, test_prep, lunch, rf_reading.feature_names_in_)

predicted_reading_score = rf_reading.predict(input_data)

print(f"Predicted Reading Score: {predicted_reading_score[0]:.2f}")

y_test_reading_pred = rf_reading.predict(X_test_reading)

reading_rmse = mean_squared_error(y_test_reading, y_test_reading_pred, squared=False)
print(f"Reading RMSE: {reading_rmse:.2f}")

reading_r2 = r2_score(y_test_reading, y_test_reading_pred)
print(f"Reading R² Score: {reading_r2:.2f}")
